from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASET_NAME = "Modotte/CodeX-2M-Thinking"
DEFAULT_OUTPUT = Path("codex_thinking_qa.json")


QUESTION_KEYS = (
    "question",
    "prompt",
    "instruction",
    "input",
    "problem",
    "query",
)
ANSWER_KEYS = (
    "answer",
    "response",
    "output",
    "completion",
    "solution",
    "messages",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(clean_text(item) for item in value if clean_text(item)).strip()
    if isinstance(value, dict):
        if "content" in value:
            return clean_text(value.get("content"))
        return json.dumps(value, ensure_ascii=False)
    return " ".join(str(value).strip().split())


def extract_from_messages(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        return "", ""
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = clean_text(message.get("content", ""))
        if not content:
            continue
        if role in {"user", "human"}:
            user_parts.append(content)
        elif role in {"assistant", "gpt", "model"}:
            assistant_parts.append(content)
    return "\n".join(user_parts).strip(), "\n".join(assistant_parts).strip()


def extract_qa(row: dict[str, Any]) -> tuple[str, str]:
    question = ""
    answer = ""

    if "messages" in row:
        question, answer = extract_from_messages(row.get("messages"))

    if not question:
        for key in QUESTION_KEYS:
            value = clean_text(row.get(key))
            if value:
                question = value
                break

    if not answer:
        for key in ANSWER_KEYS:
            if key == "messages":
                continue
            value = clean_text(row.get(key))
            if value:
                answer = value
                break

    return question.strip(), answer.strip()


def make_short_question(text: str, index: int) -> str:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line:
        return first_line[:220]
    return f"exemple de code raisonne {index}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importe une partie du dataset Modotte/CodeX-2M-Thinking dans Lucie."
    )
    parser.add_argument("--limit", type=int, default=2000, help="Nombre maximum d'exemples a importer.")
    parser.add_argument("--split", default="train", help="Split Hugging Face a lire.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Fichier JSON Lucie a creer.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Installe d'abord la dependance: python -m pip install datasets"
        ) from exc

    dataset = load_dataset(DATASET_NAME, split=args.split, streaming=True)
    examples: list[dict[str, str]] = []
    seen: set[str] = set()

    for index, row in enumerate(dataset, start=1):
        if not isinstance(row, dict):
            continue
        question, answer = extract_qa(row)
        if not answer:
            continue
        if not question:
            question = make_short_question(answer, index)
        question_key = question.lower()
        if question_key in seen:
            continue
        seen.add(question_key)
        examples.append({"question": question, "answer": answer})
        if len(examples) >= args.limit:
            break

    args.output.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OK: {len(examples)} exemples ecrits dans {args.output}")


if __name__ == "__main__":
    main()
