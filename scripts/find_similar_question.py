#!/usr/bin/env python3
"""Find the question whose LaTeX description best matches a MathML snippet.

This script demonstrates a tiny neural network pipeline that encodes MathML into
an embedding vector and compares it with candidate physics questions.  The
network is intentionally lightweight and uses only the Python standard library
so it can run in constrained environments.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

# Candidate questions provided by the UI.  The script returns whichever entry is
# most similar to the supplied MathML.
mathQuestion1 = (
    "A series $LCR$ circuit consists of $R = 80\\,\\Omega$, $X_L = 100\\,\\Omega$, "
    "and $X_C = 40\\,\\Omega$. The input voltage is $2500 \\cos(100\\pi t)\\,\\text{V}$. "
    "The amplitude of current, in the circuit, is ___$A$."
)
mathQuestion2 = (
    "A series \\(LCR\\) circuit consists of \\(R = 80\\,\\Omega\\)\\.., "
    "\\(X_L = 100\\,\\Omega\\), and \\(X_C = 40\\,\\Omega\\). The input voltage is "
    "\\(2500 \\cos(100\\pi t)\\,\\text{V}\\). The amplitude of current, in the circuit, "
    "is ___\\(A\\)."
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|\\\\[A-Za-z]+|[∑√±−=]")


@dataclass
class Vocabulary:
    """Mapping between tokens and column indices."""

    token_to_index: dict[str, int]

    @classmethod
    def build(cls, token_sequences: Iterable[Iterable[str]]) -> "Vocabulary":
        token_to_index: dict[str, int] = {}
        for sequence in token_sequences:
            for token in sequence:
                if token not in token_to_index:
                    token_to_index[token] = len(token_to_index)
        return cls(token_to_index)

    def vectorise(self, tokens: Iterable[str]) -> List[float]:
        counts = Counter(tokens)
        vector = [0.0] * len(self.token_to_index)
        for token, count in counts.items():
            index = self.token_to_index.get(token)
            if index is not None:
                vector[index] = float(count)
        return vector


class SimpleNeuralNetwork:
    """A minimal two-layer neural network for embedding vectors."""

    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 16):
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Deterministic weight initialisation using fractional constants.
        self.w1 = [[self._seed_value(r, c, input_size) for c in range(input_size)] for r in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [[self._seed_value(r + hidden_size, c, hidden_size) for c in range(hidden_size)] for r in range(output_size)]
        self.b2 = [0.0] * output_size

    @staticmethod
    def _seed_value(row: int, column: int, scale: int) -> float:
        # Quasi-random but deterministic numbers between -0.5 and 0.5.
        return ((row * 131 + column * 17) % (scale * 13 + 1)) / (scale * 13 + 1) - 0.5

    @staticmethod
    def _tanh(value: float) -> float:
        return math.tanh(value)

    def _dense(self, matrix: Sequence[Sequence[float]], bias: Sequence[float], vector: Sequence[float]) -> List[float]:
        result: List[float] = []
        for row, bias_value in zip(matrix, bias):
            activation = bias_value
            activation += sum(weight * component for weight, component in zip(row, vector))
            result.append(activation)
        return result

    def embed(self, vector: Sequence[float]) -> List[float]:
        hidden = [self._tanh(value) for value in self._dense(self.w1, self.b1, vector)]
        output = [self._tanh(value) for value in self._dense(self.w2, self.b2, hidden)]
        return output


def normalise(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [component / norm for component in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def iter_mathml_tokens(mathml: str) -> Iterable[str]:
    # Wrap the snippet to ensure valid XML before parsing.
    wrapped = f"<root>{mathml}</root>"
    root = ET.fromstring(wrapped)

    def visit(node: ET.Element) -> Iterable[str]:
        yield f"tag:{node.tag}"
        for attribute, value in sorted(node.attrib.items()):
            yield f"attr:{node.tag}:{attribute}"
            yield from TOKEN_PATTERN.findall(value)
        if node.text:
            yield from TOKEN_PATTERN.findall(node.text)
        for child in node:
            yield from visit(child)
        if node.tail:
            yield from TOKEN_PATTERN.findall(node.tail)

    return visit(root)


def iter_question_tokens(question: str) -> Iterable[str]:
    # Strip punctuation that isn't useful for matching.
    cleaned = question.replace("$", " ")
    return TOKEN_PATTERN.findall(cleaned)


def load_mathml_input(argument: str) -> str:
    potential_path = Path(argument)
    if potential_path.exists():
        return potential_path.read_text(encoding="utf-8")
    return argument


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mathml",
        nargs="?",
        help="MathML snippet or path to a file containing MathML. If omitted, the script reads from standard input.",
    )
    args = parser.parse_args(argv)

    mathml_snippet: str | None = None

    if args.mathml:
        mathml_snippet = load_mathml_input(args.mathml)
    else:
        if not sys.stdin.isatty():
            streamed = sys.stdin.read()
            if streamed.strip():
                mathml_snippet = streamed

    if not mathml_snippet:
        parser.error(
            "No MathML provided. Supply a snippet/path argument or pipe MathML to stdin."
        )

    token_sequences = [
        list(iter_mathml_tokens(mathml_snippet)),
        list(iter_question_tokens(mathQuestion1)),
        list(iter_question_tokens(mathQuestion2)),
    ]
    vocabulary = Vocabulary.build(token_sequences)
    vectors = [vocabulary.vectorise(tokens) for tokens in token_sequences]

    network = SimpleNeuralNetwork(input_size=len(vocabulary.token_to_index))

    embeddings = [normalise(network.embed(vector)) for vector in vectors]
    mathml_embedding, question1_embedding, question2_embedding = embeddings

    similarities = [
        (cosine_similarity(mathml_embedding, question1_embedding), mathQuestion1),
        (cosine_similarity(mathml_embedding, question2_embedding), mathQuestion2),
    ]
    similarities.sort(key=lambda item: item[0], reverse=True)

    print(similarities[0][1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
