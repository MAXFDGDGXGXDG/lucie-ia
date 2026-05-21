from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DEFAULT_OUTPUT = Path("gsm8k_qa.json")


def clean_answer(answer: str) -> str:
    cleaned = " ".join(str(answer).strip().split())
    if "####" in cleaned:
        reasoning, final = cleaned.rsplit("####", 1)
        final = final.strip()
        reasoning = reasoning.strip()
        if final:
            return f"{reasoning}\nReponse finale: {final}"
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Importe GSM8K dans le format questions-reponses de Lucie.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = tout importer.")
    parser.add_argument("--splits", default="train,test", help="Splits separes par des virgules.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Installe d'abord: python -m pip install datasets") from exc

    examples: list[dict[str, str]] = []
    seen: set[str] = set()
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
        for row in ds:
            question = " ".join(str(row.get("question", "")).strip().split())
            answer = clean_answer(str(row.get("answer", "")))
            if not question or not answer or question.lower() in seen:
                continue
            seen.add(question.lower())
            examples.append({"question": question, "answer": answer})
            if args.limit and len(examples) >= args.limit:
                break
        if args.limit and len(examples) >= args.limit:
            break

    args.output.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OK: {len(examples)} exemples GSM8K ecrits dans {args.output}")


if __name__ == "__main__":
    main()
