from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
DEFAULT_OUTPUT = Path("python_code_instructions_qa.json")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return "\n".join(str(value).replace("\r\n", "\n").splitlines()).strip()


def build_question(instruction: str, input_text: str) -> str:
    instruction = " ".join(instruction.strip().split())
    input_text = clean_text(input_text)
    if input_text:
        compact_input = " ".join(input_text.split())
        if len(compact_input) > 220:
            compact_input = compact_input[:220].rsplit(" ", 1)[0].rstrip() + "..."
        return f"{instruction} Entree: {compact_input}"
    return instruction


def build_answer(output: str) -> str:
    output = clean_text(output)
    if not output:
        return ""
    if "```" in output:
        return output
    return f"Voici une solution Python:\n```python\n{output}\n```"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importe iamtarun/python_code_instructions_18k_alpaca dans Lucie."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = tout importer.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Installe d'abord: python -m pip install datasets") from exc

    dataset = load_dataset(DATASET_NAME, split="train")
    examples: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in dataset:
        if not isinstance(row, dict):
            continue
        instruction = clean_text(row.get("instruction"))
        output = build_answer(clean_text(row.get("output")))
        if not instruction or not output:
            continue
        question = build_question(instruction, clean_text(row.get("input")))
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        examples.append({"question": question, "answer": output})
        if args.limit and len(examples) >= args.limit:
            break

    args.output.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OK: {len(examples)} exemples Python ecrits dans {args.output}")


if __name__ == "__main__":
    main()
