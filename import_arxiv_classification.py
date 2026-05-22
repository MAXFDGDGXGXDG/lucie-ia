from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DATASET_NAME = "ccdv/arxiv-classification"
DATASET_CONFIG = "default"
DEFAULT_OUTPUT = Path("arxiv_classification_fr_qa.json")

LABELS = [
    "math.AC",
    "cs.CV",
    "cs.AI",
    "cs.SY",
    "math.GR",
    "cs.CE",
    "cs.PL",
    "cs.IT",
    "cs.DS",
    "cs.NE",
    "math.ST",
]

LABEL_DESCRIPTIONS = {
    "math.AC": "mathematiques, algebre commutative",
    "cs.CV": "informatique, vision par ordinateur",
    "cs.AI": "informatique, intelligence artificielle",
    "cs.SY": "informatique, systemes et controle",
    "math.GR": "mathematiques, theorie des groupes",
    "cs.CE": "informatique, ingenierie informatique",
    "cs.PL": "informatique, langages de programmation",
    "cs.IT": "informatique et theorie de l'information",
    "cs.DS": "informatique, structures de donnees et algorithmes",
    "cs.NE": "informatique, reseaux neuronaux et apprentissage evolutif",
    "math.ST": "mathematiques, statistiques",
}


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    replacements = {
        "â€”": "-",
        "â€“": "-",
        "âˆ—": "*",
        "Î£": "Sigma",
        "Ã—": "x",
        "Â": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def compact(text: str, limit: int) -> str:
    text = " ".join(clean_text(text).split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def label_name(value: Any) -> str:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return str(value)
    if 0 <= index < len(LABELS):
        return LABELS[index]
    return str(value)


def extract_title_and_abstract(text: str) -> tuple[str, str]:
    cleaned = clean_text(text)
    before_abstract, sep, after_abstract = cleaned.partition("Abstract")
    title_lines = [
        line.strip()
        for line in before_abstract.splitlines()
        if line.strip() and not line.lower().startswith("arxiv:")
    ]
    title = " ".join(title_lines[:2]).strip()
    if not title:
        title = compact(cleaned.splitlines()[0] if cleaned.splitlines() else "article arxiv", 120)

    abstract = after_abstract if sep else cleaned
    abstract = re.split(r"\n\s*(?:1\s+Introduction|I\.\s+INTRODUCTION|Keywords)\b", abstract, maxsplit=1)[0]
    abstract = compact(abstract, 850)
    return compact(title, 180), abstract


def build_answer(label: str, title: str, abstract: str) -> str:
    description = LABEL_DESCRIPTIONS.get(label, "categorie scientifique arXiv")
    return (
        f"Classe probable: {label} ({description}).\n"
        f"Titre: {title}\n"
        "Pourquoi: le resume contient des indices de vocabulaire et de probleme qui correspondent a cette categorie.\n"
        f"Resume court: {abstract}"
    )


def build_examples(row: dict[str, Any]) -> list[dict[str, str]]:
    label = label_name(row.get("label"))
    title, abstract = extract_title_and_abstract(str(row.get("text", "")))
    if not abstract:
        return []
    answer = build_answer(label, title, abstract)
    return [
        {
            "question": f"Classe cet article arXiv en francais. Titre: {title}. Resume: {compact(abstract, 420)}",
            "answer": answer,
        },
        {
            "question": f"Quel est le domaine scientifique de cet article: {title} ?",
            "answer": answer,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importe un echantillon ccdv/arxiv-classification en QA francais pour Lucie."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=1500, help="Nombre maximum de QA.")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Installe d'abord: python -m pip install datasets") from exc

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=args.split, streaming=True)
    examples: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in dataset:
        if not isinstance(row, dict):
            continue
        for example in build_examples(row):
            key = example["question"].lower()
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
            if len(examples) >= args.limit:
                break
        if len(examples) >= args.limit:
            break

    if not examples:
        raise SystemExit("Aucun exemple arXiv importe.")

    args.output.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OK: {len(examples)} exemples arXiv ecrits dans {args.output}")


if __name__ == "__main__":
    main()
