#!/usr/bin/env python3
"""Find previously seen questions that are closest to a new prompt."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

from question_matcher import (
    ModelParameters,
    Vocabulary,
    cosine_similarity,
    encode_text,
    load_model,
    tokenize_text,
)


def read_query(argument: str | None) -> str:
    if argument:
        return argument
    if sys.stdin.isatty():
        print("Paste the question text and finish with Ctrl-D (Unix) or Ctrl-Z (Windows):", file=sys.stderr)
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No question provided. Pass it as an argument or via stdin.")
    return data


def build_embedding(query: str, vocabulary: Vocabulary, params: ModelParameters) -> np.ndarray:
    tokens = tokenize_text(query)
    if not tokens:
        raise SystemExit("Unable to tokenise the question text. Please provide more descriptive input.")
    embedding = encode_text(tokens, vocabulary, params)
    norm = np.linalg.norm(embedding)
    if norm == 0.0:
        return embedding
    return embedding / norm


def compute_similarities(
    query_embedding: np.ndarray, questions: Sequence[dict], top_k: int
) -> List[tuple[float, dict]]:
    scored: List[tuple[float, dict]] = []
    for record in questions:
        embedding = np.asarray(record.get("embedding", []), dtype=np.float32)
        if embedding.size == 0:
            continue
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            continue
        embedding = embedding / norm
        score = cosine_similarity(query_embedding, embedding)
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        help="Question text to compare against the trained dataset. If omitted, the script reads stdin.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/question_embeddings.json"),
        help="Path to the trained model exported by train_question_matcher.py.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of similar questions to display.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}. Train a model with train_question_matcher.py first.")

    vocabulary, params, questions = load_model(args.model)
    if not questions:
        raise SystemExit("The model file does not contain any stored question embeddings.")

    query_text = read_query(args.question)
    query_embedding = build_embedding(query_text, vocabulary, params)
    results = compute_similarities(query_embedding, questions, args.top)
    if not results:
        print("No comparable questions found in the model.")
        return 0

    for score, record in results:
        print(f"Similarity: {score:.4f} | ID: {record.get('id')}\n{record.get('text')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
