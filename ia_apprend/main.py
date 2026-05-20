from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .ai_bot import LearningBot
    from .web_app import run_web_server
except ImportError:  # pragma: no cover - script fallback
    from ai_bot import LearningBot
    from web_app import run_web_server


ROOT = Path(__file__).resolve().parent
MEMORY_FILE = ROOT / "memory.json"


def print_help() -> None:
    print("Commandes:")
    print("  /teach question | reponse")
    print("  /list")
    print("  /help")
    print("  /exit")


def parse_teach_command(text: str) -> tuple[str, str]:
    payload = text[len("/teach") :].strip()
    if "|" not in payload:
        raise ValueError("Format attendu: /teach question | reponse")
    question, answer = [part.strip() for part in payload.split("|", 1)]
    if not question or not answer:
        raise ValueError("Question et reponse ne doivent pas etre vides.")
    return question, answer


def console_main() -> None:
    try:
        bot = LearningBot.load(MEMORY_FILE)
    except Exception as exc:
        print(f"Erreur au demarrage: {exc}")
        sys.exit(1)

    if bot.startup_warning:
        print(f"Avertissement: {bot.startup_warning}")

    print("IA Apprend est prete. Tape /help pour voir les commandes.")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            try:
                bot.save()
            except OSError as exc:
                print(f"Erreur de sauvegarde: {exc}")
            print("Memoire enregistree. Au revoir.")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/list":
            knowledge = bot.list_knowledge()
            if not knowledge:
                print("Aucune connaissance enregistree.")
            else:
                for question, answer in knowledge:
                    print(f"- {question} -> {answer}")
            continue

        if user_input.startswith("/teach"):
            try:
                question, answer = parse_teach_command(user_input)
                bot.teach(question, answer)
                bot.save()
                print("Bien recu, j'ai appris ca.")
            except ValueError as exc:
                print(f"Erreur: {exc}")
            except OSError as exc:
                print(f"Erreur de sauvegarde: {exc}")
            continue

        response = bot.answer(user_input)
        print(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="IA Apprend")
    parser.add_argument(
        "--mode",
        choices=("web", "cli"),
        default="web",
        help="Lance l'interface web ou l'interface console.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Adresse d'ecoute pour l'interface web.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port pour l'interface web.",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        console_main()
        return

    run_web_server(host=args.host, port=args.port, memory_path=MEMORY_FILE)


if __name__ == "__main__":
    main()

