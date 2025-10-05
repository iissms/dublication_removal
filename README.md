# Question Deduplication Toolkit

This repository contains a React front-end for reviewing questions and a set of
Python utilities that help detect duplicate questions stored in the
`examtech` MySQL database. The utilities fetch the questions, train a compact
neural embedding model, and surface previously seen questions that most closely
match a new prompt.

The sections below describe how to prepare your environment and how to use the
training and lookup scripts with large question sets (the production database
contains about two million rows).

## Prerequisites

1. **Python** – Python 3.9 or newer is recommended.
2. **Node.js** – Required for the MathJax worker that converts LaTeX to MathML.
   Ensure `node` is available on your PATH and install the project dependencies:

   ```bash
   npm install
   ```

3. **Python packages** – Install the required libraries in the environment where
   you will run the scripts:

   ```bash
   pip install numpy pymysql
   ```

4. **Database access** – Ensure the host running the scripts can reach the
   MySQL server at `194.238.23.60` and that the credentials defined in the
   scripts are valid. Adjust them locally if your environment requires a
   different configuration.

> **Security note:** The scripts embed database credentials that were provided
> for development. Rotate the password or load it from environment variables if
> you commit customised versions of these utilities elsewhere.

## Training embeddings (`scripts/train_question_matcher.py`)

This script streams question data from the database, normalises the LaTeX
fragments with the same logic used in the React front-end, converts the result
to MathML through MathJax, builds a token vocabulary from the MathML markup,
trains a small autoencoder to produce dense embeddings, and saves the result to
`data/question_embeddings.json` by default. The training loop vectorises
questions lazily, so the peak RAM usage stays close to the batch size even on
large corpora.

### Basic usage

```bash
python3 scripts/train_question_matcher.py --limit 50000
```

Key options:

- `--limit` – Restrict the number of questions pulled from the database. Start
  with a manageable limit while validating the workflow. Even though batches are
  now built lazily, training on the full ~2M question set still demands time and
  disk space for the exported embeddings, so scale gradually.
- `--chunk-size` – Controls how many rows are streamed from MySQL per roundtrip
  (default `5000`). Increase it if you have ample memory and want faster
  transfers; decrease it if you see memory pressure.
- `--text-columns` – Override the columns that are concatenated to form the
  training text. By default the script only uses `pre_question_text` and the
  four option fields. It automatically ignores columns that are absent. Example:

  ```bash
  python3 scripts/train_question_matcher.py \
      --text-columns pre_question_text option1_text option2_text option3_text option4_text \
      --limit 100000
  ```

- `--hidden-size`, `--embedding-size`, `--learning-rate`, `--epochs`,
  `--batch-size`, `--seed` – Tune the neural network architecture and training
  loop. The default learning rate (`0.001`) is intentionally conservative to
  keep the optimiser stable on very large vocabularies.
- `--max-grad-norm`, `--weight-clip` – Additional safety valves that clamp
  gradient norms and raw weights after each update. The defaults are tuned to
  prevent the overflows observed when training on six-figure batches without
  requiring manual intervention.
- `--output` – Location of the JSON model artefact.

The generated model records metadata (column selection and number of questions)
so downstream scripts know how the embeddings were produced.

## Looking up similar questions (`scripts/find_similar_question.py`)

Once a model is trained, use the lookup script to compare a new prompt against
all stored embeddings. The tool applies the same LaTeX normalisation and MathML
conversion pipeline as the training step before embedding the query, so the
tokenisation remains consistent:

```bash
python3 scripts/find_similar_question.py "Your new question prompt here"
```

If you omit the question text, the script prompts you to paste it in stdin.
Additional flags:

- `--model` – Path to the JSON file produced by the training step (defaults to
  `data/question_embeddings.json`).
- `--top` – Number of similar questions to list (defaults to `5`).

The output shows the cosine similarity score along with the stored question ID
and combined text so you can decide whether to mark the new prompt as a
duplicate.

## Workflow summary

1. Install Python dependencies.
2. Run `scripts/train_question_matcher.py`, starting with a conservative limit.
3. Inspect the generated `data/question_embeddings.json` and confirm it
   contains a healthy sample of the database.
4. Use `scripts/find_similar_question.py` to check incoming questions against
   the trained embeddings before inserting them into the database.
5. Re-train periodically to refresh embeddings as the question bank evolves.

## Front-end (optional)

The React application in `src/` remains available if you need to visualise or
interact with question content in the browser. Standard `npm install` and
`npm start` commands supplied by Create React App continue to work.

## Troubleshooting

- **Missing columns** – If the database schema changes, supply the available
  text columns with `--text-columns`. The script will warn you about any
  missing fields instead of failing.
- **MathJax conversion errors** – Both training and lookup spin up
  `scripts/mathjax_worker.js`. Run `npm install` to ensure `mathjax-full` is
  available and verify that Node.js is on your PATH if you see conversion
  failures.
- **Performance** – Training on millions of questions is resource intensive.
  Increase the limit gradually, and consider provisioning a machine with
  sufficient CPU cores and RAM (tens of gigabytes) for full-corpus runs.
- **Model size** – The JSON file grows with the number of questions. Compress or
  rotate older snapshots if disk usage becomes an issue.

Keeping an up-to-date embedding model lets you flag duplicates quickly and
maintain a high-quality question bank without manual inspection of every entry.
