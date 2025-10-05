#!/usr/bin/env python3
"""Find previously seen questions that are closest to a new prompt."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

from question_matcher import (
    MathMLConverter,
    ModelParameters,
    Vocabulary,
    convert_text_to_mathml,
    cosine_similarity,
    encode_text,
    load_model,
    normalise_latex,
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


def build_embedding(
    query: str,
    vocabulary: Vocabulary,
    params: ModelParameters,
    converter: MathMLConverter,
) -> np.ndarray:
    normalised = normalise_latex(query)
    try:
        mathml = convert_text_to_mathml(normalised, converter)
    except Exception as exc:
        print(f"Warning: Falling back to normalised text after MathML conversion failure: {exc}", file=sys.stderr)
        mathml = normalised
    tokens = tokenize_text(mathml)
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
    try:
        with MathMLConverter() as converter:
            query_embedding = build_embedding(query_text, vocabulary, params, converter)
    except FileNotFoundError as exc:
        raise SystemExit(
            "MathJax worker script not found. Ensure scripts/mathjax_worker.js exists and dependencies are installed."
        ) from exc
    results = compute_similarities(query_embedding, questions, args.top)
    if not results:
        print("No comparable questions found in the model.")
        return 0

    for score, record in results:
        print(f"Similarity: {score:.4f} | ID: {record.get('id')}\n{record.get('text')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
