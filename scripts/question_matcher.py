"""Utilities for training and using a lightweight neural question matcher."""
from __future__ import annotations

import html
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|\\\\[A-Za-z]+|[∑√±−=]")
INLINE_DISPLAY_PATTERN = re.compile(r"([^\n])\\\[(.*?)\\\]([^\n])", re.DOTALL)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MATH_SEGMENT_PATTERN = re.compile(
    r"(\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{.*?\}.*?\\end\{.*?\}|\$\$.*?\$\$)",
    re.DOTALL,
)


def normalise_latex(text: str) -> str:
    """Replicate the front-end newline normaliser for server-side processing."""

    if not isinstance(text, str):
        return text  # type: ignore[return-value]

    out = text.replace("\r\n", "\n").replace("\\r\\n", "\n")

    replacements = [
        ("\\\\[", "\\["),
        ("\\\\]", "\\]"),
        ("\\\\(", "\\("),
        ("\\\\)", "\\)"),
    ]
    for needle, repl in replacements:
        out = out.replace(needle, repl)

    out = out.replace("\\\\", "\\")

    def _inline_replacer(match: re.Match[str]) -> str:
        left, inner, right = match.groups()
        return f"{left}\\({inner}\\){right}"

    out = INLINE_DISPLAY_PATTERN.sub(_inline_replacer, out)
    return out


def normalise_whitespace(text: str) -> str:
    """Collapse whitespace and normalise newlines for stable tokenisation."""
    return " ".join(text.replace("\r", "\n").split())


def tokenize_text(text: str) -> List[str]:
    cleaned = HTML_TAG_PATTERN.sub(" ", text)
    cleaned = normalise_whitespace(cleaned)
    cleaned = cleaned.replace("$", " ").replace("\\n", " ").lower()
    return TOKEN_PATTERN.findall(cleaned)


class MathMLConverter:
    """Convert LaTeX-rich text to MathML via a persistent MathJax worker."""

    def __init__(self, worker_path: str | Path | None = None) -> None:
        self.worker_path = Path(worker_path) if worker_path else Path(__file__).with_name("mathjax_worker.js")
        self._proc: subprocess.Popen[str] | None = None
        self._counter = 0

    def __enter__(self) -> "MathMLConverter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - cleanup
        self.close()

    def open(self) -> None:
        if self._proc is not None:
            return
        if not self.worker_path.exists():
            raise FileNotFoundError(f"MathJax worker not found at {self.worker_path}")
        self._proc = subprocess.Popen(
            ["node", str(self.worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except Exception:  # pragma: no cover - best effort
                pass
        try:
            self._proc.wait(timeout=2)
        except Exception:  # pragma: no cover - best effort
            self._proc.kill()
        finally:
            self._proc = None

    def convert(self, text: str, display: bool = False) -> str:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            self.open()
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        self._counter += 1
        payload = json.dumps({"id": self._counter, "text": text, "display": bool(display)})
        try:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError("Unable to communicate with MathJax worker") from exc

        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("MathJax worker terminated unexpectedly")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if response.get("id") != self._counter:
                continue
            if "mathml" in response:
                return response["mathml"]
            raise RuntimeError(response.get("detail") or "MathML conversion failed")


MATH_WRAPPER_PATTERN = re.compile(r"^\s*<math[^>]*>(.*)</math>\s*$", re.DOTALL)


def strip_math_delimiters(segment: str) -> tuple[str, bool]:
    trimmed = segment.strip()
    if trimmed.startswith("\\[") and trimmed.endswith("\\]"):
        return trimmed[2:-2], True
    if trimmed.startswith("$$") and trimmed.endswith("$$"):
        return trimmed[2:-2], True
    if trimmed.startswith("\\(") and trimmed.endswith("\\)"):
        return trimmed[2:-2], False
    return trimmed, trimmed.startswith("\\begin")


def strip_mathml_wrapper(mathml: str) -> str:
    match = MATH_WRAPPER_PATTERN.match(mathml)
    if match:
        return match.group(1).strip()
    return mathml.strip()


def plain_text_to_mathml(text: str) -> str:
    fragments: List[str] = []
    for chunk in re.split(r"(\n)", text):
        if not chunk:
            continue
        if chunk == "\n":
            fragments.append('<mspace linebreak="newline"/>')
            continue
        escaped = html.escape(chunk)
        fragments.append(f"<mtext>{escaped}</mtext>")
    return "".join(fragments)


def convert_text_to_mathml(text: str, converter: MathMLConverter) -> str:
    if not text.strip():
        return ""

    segments: List[tuple[str, str]] = []
    position = 0
    for match in MATH_SEGMENT_PATTERN.finditer(text):
        start, end = match.span()
        if start > position:
            segments.append(("text", text[position:start]))
        segments.append(("math", match.group(0)))
        position = end
    if position < len(text):
        segments.append(("text", text[position:]))

    parts: List[str] = []
    for kind, content in segments:
        if not content:
            continue
        if kind == "text":
            parts.append(plain_text_to_mathml(content))
            continue
        math_tex, display = strip_math_delimiters(content)
        if not math_tex.strip():
            continue
        mathml = converter.convert(math_tex, display=display)
        inner = strip_mathml_wrapper(mathml)
        if display:
            parts.append(f"<mrow>{inner}</mrow>")
        else:
            parts.append(inner)

    combined = "".join(parts).strip()
    if not combined:
        return ""
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>{combined}</mrow></math>'


@dataclass
class Vocabulary:
    token_to_index: dict[str, int]

    @classmethod
    def build(cls, token_sequences: Iterable[Sequence[str]]) -> "Vocabulary":
        token_to_index: dict[str, int] = {}
        for sequence in token_sequences:
            for token in sequence:
                if token not in token_to_index:
                    token_to_index[token] = len(token_to_index)
        return cls(token_to_index)

    @property
    def size(self) -> int:
        return len(self.token_to_index)

    def vectorise(self, tokens: Sequence[str]) -> np.ndarray:
        vector = np.zeros(self.size, dtype=np.float32)
        for token in tokens:
            index = self.token_to_index.get(token)
            if index is not None:
                vector[index] += 1.0
        total = vector.sum()
        if total > 0:
            vector /= total
        return vector

    def to_json(self) -> dict:
        return {"token_to_index": self.token_to_index}

    @classmethod
    def from_json(cls, data: dict) -> "Vocabulary":
        return cls(token_to_index=dict(data["token_to_index"]))


@dataclass
class ModelParameters:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray

    def to_json(self) -> dict:
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3.tolist(),
        }

    @classmethod
    def from_json(cls, data: dict) -> "ModelParameters":
        return cls(
            w1=np.asarray(data["w1"], dtype=np.float32),
            b1=np.asarray(data["b1"], dtype=np.float32),
            w2=np.asarray(data["w2"], dtype=np.float32),
            b2=np.asarray(data["b2"], dtype=np.float32),
            w3=np.asarray(data["w3"], dtype=np.float32),
            b3=np.asarray(data["b3"], dtype=np.float32),
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class Autoencoder:
    """A tiny fully connected autoencoder for learning dense embeddings."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        embedding_size: int = 64,
        learning_rate: float = 0.01,
        seed: int = 42,
        max_grad_norm: float = 5.0,
        weight_clip: float = 5.0,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)
        self.max_grad_norm = max_grad_norm if max_grad_norm and max_grad_norm > 0 else None
        self.weight_clip = weight_clip if weight_clip and weight_clip > 0 else None
        scale = 1.0 / math.sqrt(max(1, input_size))
        self.w1 = self._rng.normal(0.0, scale, size=(input_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = self._rng.normal(0.0, scale, size=(hidden_size, embedding_size)).astype(np.float32)
        self.b2 = np.zeros(embedding_size, dtype=np.float32)
        self.w3 = self._rng.normal(0.0, scale, size=(embedding_size, input_size)).astype(np.float32)
        self.b3 = np.zeros(input_size, dtype=np.float32)

    def _forward(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = np.clip(batch @ self.w1 + self.b1, -10.0, 10.0)
        a1 = np.tanh(z1)
        z2 = np.clip(a1 @ self.w2 + self.b2, -10.0, 10.0)
        a2 = np.tanh(z2)
        z3 = np.clip(a2 @ self.w3 + self.b3, -10.0, 10.0)
        recon = np.tanh(z3)
        return a1, a2, recon

    def fit(self, inputs, epochs: int = 50, batch_size: int = 64) -> None:
        """Train the autoencoder on either an ndarray or a lazy dataset."""

        if isinstance(inputs, np.ndarray):
            if inputs.size == 0:
                return
            n_samples = inputs.shape[0]
            for _ in range(epochs):
                permutation = self._rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    indices = permutation[start : start + batch_size]
                    batch = inputs[indices]
                    self._train_batch(batch)
            return

        if not hasattr(inputs, "vectors_for_indices"):
            raise TypeError("inputs must be either a numpy array or provide a vectors_for_indices() method")

        n_samples = len(inputs)
        if n_samples == 0:
            return

        for _ in range(epochs):
            permutation = self._rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_vectors = inputs.vectors_for_indices(batch_indices.tolist())
                self._train_batch(batch_vectors)

    def _train_batch(self, batch: np.ndarray) -> None:
        hidden, embedding, reconstruction = self._forward(batch)
        error = reconstruction - batch
        delta3 = error * (1.0 - reconstruction**2)
        grad_w3 = embedding.T @ delta3 / len(batch)
        grad_b3 = delta3.mean(axis=0)
        delta2 = (delta3 @ self.w3.T) * (1.0 - embedding**2)
        grad_w2 = hidden.T @ delta2 / len(batch)
        grad_b2 = delta2.mean(axis=0)
        delta1 = (delta2 @ self.w2.T) * (1.0 - hidden**2)
        grad_w1 = batch.T @ delta1 / len(batch)
        grad_b1 = delta1.mean(axis=0)

        grads = [
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            grad_w3,
            grad_b3,
        ]
        if self.max_grad_norm:
            total_norm = math.sqrt(
                sum(float(np.sum(np.asarray(g, dtype=np.float64) ** 2)) for g in grads)
            )
        else:
            total_norm = 0.0
        if self.max_grad_norm and total_norm > 0 and total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-8)
            grad_w1 *= scale
            grad_b1 *= scale
            grad_w2 *= scale
            grad_b2 *= scale
            grad_w3 *= scale
            grad_b3 *= scale

        self.w3 -= self.learning_rate * grad_w3
        self.b3 -= self.learning_rate * grad_b3
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1

        if self.weight_clip is not None:
            np.clip(self.w1, -self.weight_clip, self.weight_clip, out=self.w1)
            np.clip(self.w2, -self.weight_clip, self.weight_clip, out=self.w2)
            np.clip(self.w3, -self.weight_clip, self.weight_clip, out=self.w3)

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.tanh(np.clip(inputs @ self.w1 + self.b1, -10.0, 10.0))
        embedding = np.tanh(np.clip(hidden @ self.w2 + self.b2, -10.0, 10.0))
        return embedding

    def encode_dataset(self, dataset, batch_size: int = 1024) -> np.ndarray:
        if not hasattr(dataset, "vectors_for_indices"):
            raise TypeError("dataset must provide a vectors_for_indices() method")
        total = len(dataset)
        embeddings = np.empty((total, self.embedding_size), dtype=np.float32)
        position = 0
        while position < total:
            end = min(position + batch_size, total)
            indices = list(range(position, end))
            batch_vectors = dataset.vectors_for_indices(indices)
            embeddings[position:end] = self.encode(batch_vectors)
            position = end
        return embeddings

    def parameters(self) -> ModelParameters:
        return ModelParameters(self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)

    def load_parameters(self, params: ModelParameters) -> None:
        self.w1 = params.w1.copy()
        self.b1 = params.b1.copy()
        self.w2 = params.w2.copy()
        self.b2 = params.b2.copy()
        self.w3 = params.w3.copy()
        self.b3 = params.b3.copy()


def save_model(
    path: str | Path,
    vocabulary: Vocabulary,
    params: ModelParameters,
    questions: Iterable[dict],
    embeddings: np.ndarray,
    metadata: dict | None = None,
) -> None:
    payload = {
        "vocabulary": vocabulary.to_json(),
        "network": {
            "input_size": vocabulary.size,
            "hidden_size": int(params.w1.shape[1]),
            "embedding_size": int(params.w2.shape[1]),
            "weights": params.to_json(),
        },
        "questions": [],
    }
    if metadata:
        payload["metadata"] = metadata
    for record, embedding in zip(questions, embeddings.tolist()):
        payload["questions"].append({
            "id": record["id"],
            "text": record["text"],
            "embedding": embedding,
        })
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(path: str | Path) -> tuple[Vocabulary, ModelParameters, List[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vocabulary = Vocabulary.from_json(data["vocabulary"])
    params = ModelParameters.from_json(data["network"]["weights"])
    questions: List[dict] = data["questions"]
    return vocabulary, params, questions


def encode_text(tokens: Sequence[str], vocabulary: Vocabulary, params: ModelParameters) -> np.ndarray:
    vector = vocabulary.vectorise(tokens)
    hidden = np.tanh(vector @ params.w1 + params.b1)
    embedding = np.tanh(hidden @ params.w2 + params.b2)
    return embedding
