from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .ai_bot import LearningBot


CASES: list[tuple[str, tuple[str, ...]]] = [
    ("bonjour", ("Bonjour", "pre")),
    ("combien font 12+30", ("42",)),
    ("combien font 9*9", ("81",)),
    ("corrige ca", ("phrase", "corriger")),
    ("resume ce texte", ("texte", "resumer", "resume")),
    ("explique une IA simplement", ("IA", "intelligence", "sujet")),
    ("qui es tu", ("Lucie",)),
    ("merci", ("plaisir",)),
    ("qu'est-ce qu'un serveur ?", ("serveur", "service", "ordinateur")),
    ("fais un plan", ("texte", "sujet")),
    ("quiz", ("sujet",)),
    ("traduis", ("texte",)),
    ("parle moi de python", ("python", "langage")),
    ("c'est quoi une variable", ("variable", "valeur")),
    ("c'est quoi une boucle", ("boucle", "repete", "repeter")),
    ("c'est quoi une fonction", ("fonction",)),
    ("explique html", ("html", "web")),
    ("explique css", ("css", "style")),
    ("javascript", ("javascript", "web", "code")),
    ("qu'est-ce que la photosynthese", ("plante", "lumiere", "energie")),
    ("qui etait napoleon", ("napoleon", "france", "empereur")),
    ("c'est quoi une fraction", ("fraction", "part")),
    ("calcule 100/4", ("25",)),
    ("calcule 7+8*2", ("23",)),
    ("aide moi", ("question", "calcul", "explication")),
    ("je ne comprends pas", ("precise", "exemple", "sujet")),
    ("quoi", ("preciser", "contexte", "sujet")),
    ("resume notre discussion", ("discussion", "contexte", "Sujet")),
    ("donne un exemple", ("exemple", "sujet")),
    ("explique python plus simple", ("python", "simple")),
    ("continue sur python", ("python", "suite", "contin")),
    ("c'est quoi github", ("git", "code", "projet")),
    ("c'est quoi une API", ("api", "application", "communiquer")),
    ("c'est quoi un robot", ("robot", "capteur", "moteur")),
    ("comment apprendre mieux", ("apprendre", "exemple", "pratique")),
    ("donne moi une idee de projet", ("projet", "idee")),
    ("explique une batterie lipo", ("batterie", "lipo", "energie")),
    ("c'est quoi un servo", ("servo", "moteur", "angle")),
    ("a quoi sert un raspberry pi", ("raspberry", "ordinateur", "carte")),
    ("c'est quoi une camera", ("image", "video", "lumiere")),
    ("c'est quoi le bluetooth", ("bluetooth", "sans fil")),
    ("explique une carte de chambre", ("carte", "piece", "position")),
    ("comment eviter les reponses fausses", ("preciser", "corriger", "source")),
    ("comment signaler une mauvaise reponse", ("signaler", "admin", "corriger")),
    ("comment ajouter une connaissance", ("admin", "fiche", "connaissance")),
    ("comment ouvrir admin", ("admin", "code")),
    ("comment changer mon profil", ("profil", "questionnaire")),
    ("comment retrouver mes anciens chats", ("historique", "conversation")),
    ("comment rendre lucie plus rapide", ("memoire", "sauvegarde", "rapide")),
    ("au revoir", ("suite", "bientot", "OK")),
]


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_path = Path(temp_dir) / "memory.json"
        for question, expected_parts in CASES:
            bot = LearningBot.load(memory_path)
            bot.teach("qu'est-ce qu'un serveur ?", "Un serveur est un ordinateur qui fournit un service.")
            bot.teach("c'est quoi un servo", "Un servo est un moteur controle en angle.")
            bot.teach("c'est quoi une variable", "Une variable garde une valeur pour la reutiliser.")
            bot.teach("c'est quoi une boucle", "Une boucle repete une action plusieurs fois.")
            bot.teach("c'est quoi une fonction", "Une fonction regroupe des instructions reutilisables.")
            bot.teach("c'est quoi github", "GitHub sert a stocker et partager du code avec Git.")
            bot.teach("c'est quoi une fraction", "Une fraction represente une partie d'un tout.")
            bot.teach("qui etait napoleon", "Napoleon etait un empereur francais.")
            bot.teach("javascript", "JavaScript est un langage de code utilise sur le web.")
            bot.teach("explique python plus simple", "Python est un langage de code simple a lire.")
            bot.teach("continue sur python", "Suite sur Python: on peut apprendre les variables, les boucles et les fonctions.")
            bot.teach("c'est quoi une api", "Une API permet a deux applications de communiquer.")
            bot.teach("donne moi une idee de projet", "Idee de projet: cree un mini assistant avec chat, memoire et admin.")
            bot.teach("a quoi sert un raspberry pi", "Un Raspberry Pi est un petit ordinateur en carte utile pour robotique et projets.")
            bot.teach("c'est quoi une camera", "Une camera capture une image ou une video avec la lumiere.")
            bot.teach("c'est quoi le bluetooth", "Le Bluetooth est une connexion sans fil courte distance.")
            bot.teach("comment eviter les reponses fausses", "Pour eviter les reponses fausses, Lucie doit demander une precision, utiliser une source et accepter les corrections.")
            bot.teach("comment signaler une mauvaise reponse", "Pour signaler une mauvaise reponse, clique sur ameliorer cette reponse puis corrige dans l'admin.")
            bot.teach("comment ajouter une connaissance", "Pour ajouter une connaissance, ouvre l'admin et ajoute une fiche dans la base de connaissances.")
            bot.teach("comment ouvrir admin", "Pour ouvrir admin, va sur /admin et entre le code.")
            bot.teach("comment changer mon profil", "Pour changer ton profil, relance le questionnaire ou modifie tes infos utilisateur.")
            bot.teach("comment retrouver mes anciens chats", "L'historique serveur permet de retrouver les anciennes conversations.")
            bot.teach("comment rendre lucie plus rapide", "Pour rendre Lucie plus rapide, on compacte la memoire et on groupe les sauvegardes.")
            bot.teach("au revoir", "A bientot. Je garde le fil pour la suite.")
            answer = bot.answer(question)
            answer_l = answer.lower()
            if not any(part.lower() in answer_l for part in expected_parts):
                failures.append(f"{question!r} -> {answer!r}")
        memory_bot = LearningBot.load(memory_path)
        memory_bot.answer("je m'appelle Maxence")
        remembered_name = memory_bot.answer("comment je m'appelle")
        if "maxence" not in remembered_name.lower():
            failures.append(f"memory name -> {remembered_name!r}")
        memory_bot.answer("j'aime la robotique")
        remembered_like = memory_bot.answer("qu'est-ce que j'aime")
        if "robotique" not in remembered_like.lower():
            failures.append(f"memory like -> {remembered_like!r}")
    if failures:
        print("QUALITY CHECK FAILED")
        for failure in failures[:12]:
            print("-", failure)
        print(f"{len(failures)} failure(s)")
        return 1
    print(f"QUALITY CHECK OK: {len(CASES)} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
