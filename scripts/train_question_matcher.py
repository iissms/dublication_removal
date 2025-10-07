#!/usr/bin/env python3
"""Train question embeddings from the examtech database and save them locally."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import sys

import numpy as np

from question_matcher import (
    Autoencoder,
    MathMLConverter,
    Vocabulary,
    convert_text_to_mathml,
    normalise_latex,
    save_model,
    tokenize_text,
)

try:
    import pymysql
except ImportError as exc:  # pragma: no cover - import error path
    raise SystemExit(
        "pymysql is required to connect to the MySQL database. Install it with 'pip install pymysql'."
    ) from exc


DATABASE_CONFIG = {
    "host": "194.238.23.60",
    "user": "lohith_pc",
    "password": "lohith_pc",
    "database": "examtech",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "read_timeout": 30,
    "write_timeout": 30,
    "connect_timeout": 30,
}


DEFAULT_TEXT_COLUMNS = [
    "pre_question_text",
    "option1_text",
    "option2_text",
    "option3_text",
    "option4_text",
]


@dataclass
class QuestionRecord:
    id: int
    text: str
    tokens: List[str]


def fetch_questions(
    limit: int | None = None,
    chunk_size: int = 5000,
    text_columns: Sequence[str] | None = None,
    mathml_converter: MathMLConverter | None = None,
) -> Tuple[List[QuestionRecord], List[str]]:
    connection = pymysql.connect(**DATABASE_CONFIG)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW COLUMNS FROM questions")
            available_columns = {row["Field"] for row in cursor.fetchall()}

        requested_columns = list(dict.fromkeys(text_columns or DEFAULT_TEXT_COLUMNS))
        selected_columns = [column for column in requested_columns if column in available_columns]
        missing_columns = [column for column in requested_columns if column not in available_columns]
        if missing_columns:
            print(
                "Warning: the following text columns are not present in the questions table and will be ignored: "
                + ", ".join(missing_columns),
                file=sys.stderr,
            )
        if not selected_columns:
            raise SystemExit(
                "None of the requested text columns were found in the questions table. "
                "Use --text-columns to specify the available fields."
            )

        select_clause = ", ".join(["`id`"] + [f"`{column}`" for column in selected_columns])
        query = f"SELECT {select_clause} FROM questions"
        if limit is not None:
            query += " LIMIT %s"

        records: List[QuestionRecord] = []
        with connection.cursor(pymysql.cursors.SSDictCursor) as cursor:
            if limit is not None:
                cursor.execute(query, (limit,))
            else:
                cursor.execute(query)
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                for row in rows:
                    combined_parts = []
                    for column in selected_columns:
                        value = row.get(column)
                        if not value:
                            continue
                        text_value = str(value).strip()
                        if text_value:
                            combined_parts.append(text_value)
                    combined = " ".join(combined_parts).strip()
                    if not combined:
                        continue
                    normalised = normalise_latex(combined)
                    transformed = normalised
                    if mathml_converter is not None:
                        try:
                            transformed = convert_text_to_mathml(normalised, mathml_converter)
                        except Exception as exc:  # pragma: no cover - conversion guard
                            print(
                                f"Warning: MathML conversion failed for question {row['id']}: {exc}",
                                file=sys.stderr,
                            )
                            transformed = normalised
                    tokens = tokenize_text(transformed)
                    if not tokens:
                        continue
                    records.append(QuestionRecord(id=row["id"], text=transformed, tokens=tokens))
    finally:
        connection.close()

    if not records:
        raise SystemExit("No usable question rows were retrieved from the database.")
    return records, selected_columns


class QuestionDataset:
    """Vectorises question records lazily to reduce peak memory usage."""

    def __init__(self, records: Sequence[QuestionRecord], vocabulary: Vocabulary) -> None:
        self._records = list(records)
        self._vocabulary = vocabulary

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    def vectors_for_indices(self, indices: Sequence[int]) -> np.ndarray:
        batch = np.zeros((len(indices), self._vocabulary.size), dtype=np.float32)
        for row, record_index in enumerate(indices):
            tokens = self._records[record_index].tokens
            batch[row] = self._vocabulary.vectorise(tokens)
        return batch


def ensure_directory(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def train_model(
    records: Sequence[QuestionRecord],
    hidden_size: int,
    embedding_size: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    max_grad_norm: float,
    weight_clip: float,
) -> tuple[Vocabulary, Autoencoder, np.ndarray]:
    token_sequences = [record.tokens for record in records]
    vocabulary = Vocabulary.build(token_sequences)
    dataset = QuestionDataset(records, vocabulary)

    autoencoder = Autoencoder(
        input_size=vocabulary.size,
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        learning_rate=learning_rate,
        seed=seed,
        max_grad_norm=max_grad_norm,
        weight_clip=weight_clip,
    )
    autoencoder.fit(dataset, epochs=epochs, batch_size=batch_size)
    embeddings = autoencoder.encode_dataset(dataset, batch_size=max(batch_size, 1024))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return vocabulary, autoencoder, embeddings


def save_trained_model(
    output_path: Path,
    vocabulary: Vocabulary,
    autoencoder: Autoencoder,
    records: Sequence[QuestionRecord],
    embeddings: np.ndarray,
    used_columns: Sequence[str] | None = None,
) -> None:
    ensure_directory(output_path)
    questions = [
        {"id": record.id, "text": record.text}
        for record in records
    ]
    metadata = {"question_count": len(records)}
    if used_columns:
        metadata["text_columns"] = list(used_columns)
    save_model(
        output_path,
        vocabulary,
        autoencoder.parameters(),
        questions,
        embeddings,
        metadata=metadata,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions fetched from the database.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows to stream from the database at a time.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Number of neurons in the hidden layer.",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=64,
        help="Size of the embedding vector.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for gradient descent (lower values improve stability).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight initialisation.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Clip gradient norms to this value to avoid numerical instability.",
    )
    parser.add_argument(
        "--weight-clip",
        type=float,
        default=5.0,
        help="Clip network weights to this absolute value after each update.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/question_embeddings.json"),
        help="Path to the output model file.",
    )
    parser.add_argument(
        "--text-columns",
        nargs="+",
        metavar="COLUMN",
        default=None,
        help=(
            "Columns from the questions table to concatenate when building training text. "
            "Defaults to pre_question_text and the four option columns."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with MathMLConverter() as mathml_converter:
            records, used_columns = fetch_questions(
                limit=args.limit,
                chunk_size=args.chunk_size,
                text_columns=args.text_columns,
                mathml_converter=mathml_converter,
            )
    except FileNotFoundError as exc:
        raise SystemExit(
            "MathJax worker script not found. Ensure scripts/mathjax_worker.js exists and dependencies are installed."
        ) from exc

    print(
        f"Loaded {len(records)} questions using columns: {', '.join(used_columns)}",
        file=sys.stderr,
    )

    vocabulary, autoencoder, embeddings = train_model(
        records,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        weight_clip=args.weight_clip,
    )

    save_trained_model(
        args.output,
        vocabulary,
        autoencoder,
        records,
        embeddings,
        used_columns=used_columns,
    )
    print(f"Saved trained model with {len(records)} questions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
