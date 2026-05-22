from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_OUTPUT = Path("local_personal_knowledge.json")
DEFAULT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".html", ".css", ".js"}
SECRET_PATTERNS = (
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"token\s*[:=]", re.IGNORECASE),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
    re.compile(r"passwd\s*[:=]", re.IGNORECASE),
    re.compile(r"secret\s*[:=]", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".cache",
    "AppData",
}


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def looks_secret(text: str) -> bool:
    sample = text[:4000]
    return any(pattern.search(sample) for pattern in SECRET_PATTERNS)


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def read_text_file(path: Path, max_chars: int) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    raw = clean_text(raw)
    if len(raw) > max_chars:
        raw = raw[:max_chars].rsplit("\n", 1)[0].rstrip() + "\n..."
    return raw


def summarize_for_question(path: Path, text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if len(first_line) > 120:
        first_line = first_line[:120].rsplit(" ", 1)[0].rstrip() + "..."
    return first_line or path.stem


def build_examples(path: Path, root: Path, text: str) -> list[dict[str, str]]:
    relative = path.relative_to(root)
    title = str(relative).replace("\\", "/")
    hint = summarize_for_question(path, text)
    answer = (
        f"Source locale privee: {title}\n"
        f"Resume utile: {hint}\n"
        f"Contenu:\n{text}"
    )
    return [
        {
            "question": f"que sais-tu du fichier {title}",
            "answer": answer,
        },
        {
            "question": f"resume mon fichier {title}",
            "answer": answer,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importe des fichiers locaux choisis dans une base privee pour Lucie."
    )
    parser.add_argument("folders", nargs="+", type=Path, help="Dossiers a importer explicitement.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--extensions", default=",".join(sorted(DEFAULT_EXTENSIONS)))
    args = parser.parse_args()

    extensions = {
        item.strip().lower()
        for item in args.extensions.split(",")
        if item.strip().startswith(".")
    }
    examples: list[dict[str, str]] = []
    skipped_secret = 0
    skipped_other = 0

    for folder in args.folders:
        root = folder.expanduser().resolve()
        if not root.exists() or not root.is_dir():
            print(f"Ignore: dossier introuvable {root}")
            continue
        for path in root.rglob("*"):
            if len(examples) // 2 >= args.max_files:
                break
            if not path.is_file() or should_skip(path):
                continue
            if path.suffix.lower() not in extensions:
                skipped_other += 1
                continue
            text = read_text_file(path, args.max_chars)
            if not text:
                skipped_other += 1
                continue
            if looks_secret(text):
                skipped_secret += 1
                continue
            examples.extend(build_examples(path, root, text))

    if not examples:
        raise SystemExit("Aucun fichier importe. Choisis un dossier avec des fichiers texte/code lisibles.")

    args.output.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OK: {len(examples)} Q/R creees dans {args.output}")
    print(f"Fichiers importes: {len(examples) // 2}")
    print(f"Fichiers ignores car secrets probables: {skipped_secret}")
    print(f"Autres fichiers ignores: {skipped_other}")
    print("Important: ce fichier est ignore par Git et reste local.")


if __name__ == "__main__":
    main()
