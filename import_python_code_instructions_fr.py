from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
DEFAULT_OUTPUT = Path("python_code_instructions_fr_qa.json")


PHRASE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("create a function to", "cree une fonction pour"),
    ("create a python function to", "cree une fonction Python pour"),
    ("create a python function that", "cree une fonction Python qui"),
    ("create a python list comprehension to", "cree une comprehension de liste Python pour"),
    ("create a python script to", "cree un script Python pour"),
    ("create a python program to", "cree un programme Python pour"),
    ("create a program to", "cree un programme pour"),
    ("create a code to", "cree un code pour"),
    ("create a", "cree un"),
    ("generate a python code for", "genere du code Python pour"),
    ("generate a python script to", "genere un script Python pour"),
    ("generate a python program to", "genere un programme Python pour"),
    ("generate a rest api with python and flask that", "genere une API REST avec Python et Flask qui"),
    ("generate a python code to", "genere un code Python pour"),
    ("generate a", "genere un"),
    ("write a python code to", "ecris un code Python pour"),
    ("write a python script to", "ecris un script Python pour"),
    ("write a python program to", "ecris un programme Python pour"),
    ("write a python function to", "ecris une fonction Python pour"),
    ("write a function to", "ecris une fonction pour"),
    ("write a code to", "ecris un code pour"),
    ("write a", "ecris un"),
    ("develop a python program to", "developpe un programme Python pour"),
    ("develop a", "developpe un"),
    ("implement a function to", "implemente une fonction pour"),
    ("implement a", "implemente un"),
    ("design a", "concois un"),
    ("convert the following", "convertis le suivant"),
    ("convert a", "convertis un"),
    ("calculate the", "calculer le"),
    ("calculate", "calculer"),
    ("find the", "trouver le"),
    ("find all", "trouver tous les"),
    ("find", "trouver"),
    ("get the", "obtenir le"),
    ("get all", "obtenir tous les"),
    ("get", "obtenir"),
    ("return the", "retourner le"),
    ("return", "retourner"),
    ("print the", "afficher le"),
    ("print", "afficher"),
    ("remove all", "supprimer tous les"),
    ("remove the", "supprimer le"),
    ("remove", "supprimer"),
    ("sort the", "trier le"),
    ("sort", "trier"),
    ("check if", "verifier si"),
    ("check whether", "verifier si"),
    ("check", "verifier"),
    ("count the", "compter le"),
    ("count", "compter"),
    ("replace the", "remplacer le"),
    ("replace", "remplacer"),
    ("reverse the", "inverser le"),
    ("reverse", "inverser"),
    ("merge two", "fusionner deux"),
    ("merge", "fusionner"),
    ("filter the", "filtrer le"),
    ("filter", "filtrer"),
    ("validate", "valider"),
    ("parse", "analyser"),
    ("crawl", "parcourir"),
    ("crawling", "parcourir"),
    ("scrape", "extraire"),
)

WORD_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("calculate the sum of a", "calculer la somme d'une"),
    ("get the squared values of a", "obtenir les valeurs au carre d'une"),
    ("get the third largest element in a", "obtenir le troisieme plus grand element dans une"),
    ("crawling a website for a", "parcourir un site web pour un"),
    ("takes in a string and a list of words", "prend en entree une chaine de caracteres et une liste de mots"),
    ("what should this python program do", "que doit faire ce programme Python"),
    ("perform this action", "effectuer cette action"),
    ("given a string", "etant donne une chaine de caracteres"),
    ("given a list", "etant donne une liste"),
    ("that takes in", "qui prend en entree"),
    ("and returns true if", "et retourne vrai si"),
    ("contains all the words", "contient tous les mots"),
    ("specific type of data", "type precis de donnees"),
    ("data to crawl", "donnees a parcourir"),
    ("allows users to", "permet aux utilisateurs de"),
    ("sequence of integers", "sequence d'entiers"),
    ("list comprehension", "comprehension de liste"),
    ("squared values", "valeurs au carre"),
    ("given list", "liste donnee"),
    ("given row", "ligne donnee"),
    ("third largest element", "troisieme plus grand element"),
    ("random numbers", "nombres aleatoires"),
    ("random number", "nombre aleatoire"),
    ("between", "entre"),
    ("divisible by", "divisibles par"),
    ("consecutive duplicates", "doublons consecutifs"),
    ("duplicates", "doublons"),
    ("string", "chaine de caracteres"),
    ("website", "site web"),
    ("phone numbers", "numeros de telephone"),
    ("database", "base de donnees"),
    ("users", "utilisateurs"),
    ("records", "enregistrements"),
    ("create, read, update, and delete", "creer, lire, modifier et supprimer"),
    ("calculate", "calculer"),
    ("crawling", "parcourir"),
    ("takes in", "prend en entree"),
    ("takes", "prend"),
    ("generates", "generer"),
    ("generate", "generer"),
    ("returns", "retourne"),
    ("return", "retourner"),
    ("sum", "somme"),
    ("integers", "entiers"),
    ("numbers", "nombres"),
    ("element", "element"),
    ("elements", "elements"),
    ("values", "valeurs"),
    ("list", "liste"),
    ("words", "mots"),
    ("word", "mot"),
    ("dictionary", "dictionnaire"),
    ("file", "fichier"),
    ("data", "donnees"),
    ("input", "entree"),
    ("output", "sortie"),
    ("that are", "qui sont"),
    ("the following", "le suivant"),
    ("the ", "le "),
    ("from the", "depuis le"),
    ("from a", "depuis une"),
    (" of a ", " d'une "),
    (" of an ", " d'un "),
    ("for a", "pour un"),
    ("for", "pour"),
    (" of ", " de "),
    (" a ", " une "),
    (" an ", " un "),
    (" in a ", " dans une "),
    (" in the ", " dans le "),
    (" to ", " pour "),
    (" and ", " et "),
)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return "\n".join(str(value).replace("\r\n", "\n").splitlines()).strip()


def translate_instruction(instruction: str) -> str:
    text = " ".join(clean_text(instruction).split())
    if not text:
        return ""

    lowered = text.lower()
    translated = lowered
    for source, target in PHRASE_REPLACEMENTS:
        if translated.startswith(source):
            translated = target + translated[len(source):]
            break

    for source, target in WORD_REPLACEMENTS:
        if source.startswith(" ") or source.endswith(" "):
            translated = translated.replace(source, target)
        else:
            translated = re.sub(rf"\b{re.escape(source)}\b", target, translated)

    translated = translated.replace(" python ", " Python ")
    if translated.startswith("python "):
        translated = "Python " + translated[len("python "):]
    translated = translated.replace(" api ", " API ")
    translated = translated.replace(" rest ", " REST ")
    translated = translated.replace(" flask ", " Flask ")
    translated = translated.replace(" sql ", " SQL ")
    translated = translated.replace(" json ", " JSON ")
    translated = translated.replace("in le ", "dans la ")
    translated = translated.replace(" le chaine", " la chaine")
    translated = translated.replace(" le liste", " la liste")
    translated = translated.replace("generer nombres", "generer des nombres")
    translated = translated.replace("supprimer enregistrements", "supprimer des enregistrements")
    translated = translated.strip()
    if translated and translated[0].islower():
        translated = translated[0].upper() + translated[1:]
    return translated.rstrip(".?") + "."


def build_question(instruction: str, input_text: str) -> str:
    question = translate_instruction(instruction)
    input_text = clean_text(input_text)
    if input_text:
        compact_input = " ".join(input_text.split())
        compact_input = translate_input(compact_input)
        if len(compact_input) > 220:
            compact_input = compact_input[:220].rsplit(" ", 1)[0].rstrip() + "..."
        return f"{question} Entree: {compact_input}"
    return question


def translate_input(text: str) -> str:
    translated = text
    replacements = (
        ("Given a string", "Etant donne une chaine"),
        ("Given a list", "Etant donne une liste"),
        ("remove all the consecutive duplicates from the string", "supprimer tous les doublons consecutifs de la chaine"),
        ("Input:", "Entree:"),
        ("website:", "site web:"),
        ("data to crawl:", "donnees a parcourir:"),
        ("phone numbers", "numeros de telephone"),
        ("Not applicable", "Non applicable"),
    )
    for source, target in replacements:
        translated = translated.replace(source, target)
    return translated


def build_answer(output: str) -> str:
    output = clean_text(output)
    if not output:
        return ""
    if "```" in output:
        return output
    return (
        "Voici une solution Python. Le code est garde tel quel pour pouvoir etre copie et teste:\n"
        f"```python\n{output}\n```"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importe le dataset Python code instructions en questions francaises pour Lucie."
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
    print(f"OK: {len(examples)} exemples Python FR ecrits dans {args.output}")


if __name__ == "__main__":
    main()
