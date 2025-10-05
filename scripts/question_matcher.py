"""Utilities for training and using a lightweight neural question matcher."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|\\\\[A-Za-z]+|[∑√±−=]")


def normalise_whitespace(text: str) -> str:
    """Collapse whitespace and normalise newlines for stable tokenisation."""
    return " ".join(text.replace("\r", "\n").split())


def tokenize_text(text: str) -> List[str]:
    cleaned = normalise_whitespace(text)
    cleaned = cleaned.replace("$", " ").replace("\\n", " ").lower()
    return TOKEN_PATTERN.findall(cleaned)


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
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(max(1, input_size))
        self.w1 = self._rng.normal(0.0, scale, size=(input_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = self._rng.normal(0.0, scale, size=(hidden_size, embedding_size)).astype(np.float32)
        self.b2 = np.zeros(embedding_size, dtype=np.float32)
        self.w3 = self._rng.normal(0.0, scale, size=(embedding_size, input_size)).astype(np.float32)
        self.b3 = np.zeros(input_size, dtype=np.float32)

    def _forward(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = batch @ self.w1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.tanh(z2)
        z3 = a2 @ self.w3 + self.b3
        recon = np.tanh(z3)
        return a1, a2, recon

    def fit(self, inputs: np.ndarray, epochs: int = 50, batch_size: int = 64) -> None:
        if inputs.size == 0:
            return
        n_samples = inputs.shape[0]
        for epoch in range(epochs):
            permutation = self._rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                indices = permutation[start : start + batch_size]
                batch = inputs[indices]
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

                self.w3 -= self.learning_rate * grad_w3
                self.b3 -= self.learning_rate * grad_b3
                self.w2 -= self.learning_rate * grad_w2
                self.b2 -= self.learning_rate * grad_b2
                self.w1 -= self.learning_rate * grad_w1
                self.b1 -= self.learning_rate * grad_b1

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.tanh(inputs @ self.w1 + self.b1)
        embedding = np.tanh(hidden @ self.w2 + self.b2)
        return embedding

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
