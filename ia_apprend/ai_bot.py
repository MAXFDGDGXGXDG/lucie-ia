from __future__ import annotations

import ast
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from openai import OpenAI

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None


DEFAULT_MODEL = os.getenv("IA_MODEL", "gpt-5.5")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
GPT4ALL_BASE_URL = os.getenv("GPT4ALL_BASE_URL", "http://127.0.0.1:4891/v1").strip()
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "https://api.dify.ai/v1").rstrip("/")
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "").strip()
DIFY_DATASET_ID = os.getenv(
    "DIFY_DATASET_ID",
    "",
).strip()
ROBOT_BRIDGE_URL = os.getenv("MAKEBLOCK_BRIDGE_URL", "http://127.0.0.1:8765").strip().rstrip("/")
TRAINING_FILE_NAME = "label_studio_training.json"
NER_FILE_NAME = "label_studio_ner_clean.json"
RELATIONS_FILE_NAME = "label_studio_relations_clean.json"
QA_SEED_FILE_NAME = "qa_seed.json"
QA_MODEL_FILE_NAME = "qa_model.json"
SCIENCE_FILE_NAME = "science.json"
EXTRA_QA_FILE_NAMES = (
    "science.json",
    "celine dion.json",
    "histoire.json",
    "histoire deux.json",
    "PHYTON.json",
    "WEB.json",
    "mega_qa_pack.json",
    "lucie_extra_questions.json",
    "lucie_extra_questions_2.json",
    "codex_thinking_qa.json",
    "gsm8k_qa.json",
)
DEFAULT_EXAMPLES: list[dict[str, str]] = [
    {"question": "bonjour", "answer": "Bonjour !"},
    {
        "question": "comment faire",
        "answer": "Je peux donner les \u00e9tapes: entr\u00e9e, traitement, sortie, avec un exemple. Precise le sujet exact et je l'adapte.",
    },
    {
        "question": "comment tu t'appelles",
        "answer": "Je suis une IA qui apprend avec le contexte et la memoire.",
    },
]
MAX_HISTORY_TURNS = 12
def _make_french_spellchecker() -> Any:
    if SpellChecker is None:
        return None
    try:
        return SpellChecker(language="fr")
    except Exception:
        return None


FR_SPELLCHECKER = _make_french_spellchecker()
FR_WORD_KEEP = {
    "ai",
    "ia",
    "api",
    "openai",
    "gpt",
    "gpt4all",
    "python",
    "html",
    "css",
    "json",
    "web",
    "dify",
    "llm",
}
EXAMPLE_INDEX_STOP_WORDS = {
    "alors",
    "avec",
    "cette",
    "dans",
    "des",
    "donc",
    "elle",
    "est",
    "ils",
    "les",
    "mes",
    "mon",
    "ont",
    "par",
    "pas",
    "pour",
    "que",
    "qui",
    "quoi",
    "son",
    "sur",
    "tes",
    "ton",
    "une",
    "vous",
}

MATH_LEAD_INS = (
    "calcule",
    "calcul",
    "combien font",
    "combien fait",
    "que vaut",
    "resous",
    "r?sous",
    "resoudre",
    "r?soudre",
    "fais le calcul de",
    "fait le calcul de",
    "donne-moi le resultat de",
    "donne moi le resultat de",
)

MATH_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "pow": pow,
    "ceil": math.ceil,
    "floor": math.floor,
}

SYNTHETIC_RESPONSE_MODES = ("court", "clair", "detaille", "debutant", "exemple")
SYNTHETIC_QUESTION_KINDS = (
    "definition",
    "utilite",
    "pourquoi",
    "comment",
    "exemple",
    "avantage",
    "risque",
    "comparaison",
    "relation",
    "resume",
    "court",
    "detaille",
    "explication",
    "fonctionnement",
    "limite",
    "etapes",
    "usage",
    "conseil",
    "erreur",
    "situation",
)
SYNTHETIC_VARIANTS_PER_PAIR = 1_000_000
SYNTHETIC_TOPIC_FACTS: dict[str, dict[str, object]] = {
    "python": {
        "keywords": ("python", "pyhton", "script", "programme"),
        "definition": "Python est un langage de programmation simple a lire et tres polyvalent.",
        "utilite": "On l'utilise pour automatiser, analyser des donnees, faire du web et de l'IA.",
        "pourquoi": "Il est apprecie parce qu'il est lisible, rapide a prendre en main et tres utilise.",
        "comment": "Pour commencer, on ecrit une variable, une condition ou une boucle tres simple.",
        "exemple": "Exemple : un petit script qui renomme des fichiers ou affiche un message.",
    },
    "web": {
        "keywords": ("web", "html", "css", "site", "internet"),
        "definition": "Le web est l'ensemble des pages et services accessibles dans un navigateur.",
        "utilite": "On l'utilise pour afficher des contenus, proposer des services et creer des applications.",
        "pourquoi": "Il est partout parce qu'il permet de partager facilement des informations.",
        "comment": "On combine souvent HTML, CSS, JavaScript et une API pour creer une app web.",
        "exemple": "Exemple : un site avec un champ de recherche et des resultats dynamiques.",
    },
    "ia": {
        "keywords": ("ia", "intelligence artificielle", "llm", "modele"),
        "definition": "L'IA est un ensemble de techniques qui aident une machine a reconnaitre, predire et repondre.",
        "utilite": "On l'utilise pour discuter, resumer, classer, traduire et retrouver de l'information.",
        "pourquoi": "Elle est utile parce qu'elle automatise des taches et aide a traiter beaucoup de texte.",
        "comment": "Elle apprend a partir de donnees, d'exemples et parfois d'un retour utilisateur.",
        "exemple": "Exemple : un assistant qui repond a une question et resume un document.",
    },
    "serveur": {
        "keywords": ("serveur", "backend", "api", "service"),
        "definition": "Un serveur est une machine ou un programme qui fournit un service a d'autres applications.",
        "utilite": "On l'utilise pour stocker des donnees, gerer des requetes et repondre a des clients.",
        "pourquoi": "Il est utile parce qu'il centralise le traitement et rend le service accessible.",
        "comment": "Un client envoie une requete, le serveur traite la demande, puis renvoie une reponse.",
        "exemple": "Exemple : une API qui renvoie une liste de documents ou une reponse de chat.",
    },
    "document": {
        "keywords": ("document", "texte", "fichier", "note", "pdf"),
        "definition": "Un document est un contenu organise qui contient de l'information a lire ou a exploiter.",
        "utilite": "On l'utilise pour conserver des cours, des consignes, des resumes ou des donnees.",
        "pourquoi": "Il est utile parce qu'il garde une trace propre d'une information importante.",
        "comment": "On peut le decouper en passages pour retrouver plus vite la bonne information.",
        "exemple": "Exemple : un cours de science, une fiche de revision ou un mode d'emploi.",
    },
    "memoire": {
        "keywords": ("memoire", "souvenir", "retenir", "historique"),
        "definition": "La memoire est ce que l'IA garde pour se souvenir d'un fait, d'un contexte ou d'une preference.",
        "utilite": "On l'utilise pour retrouver le fil d'une conversation et personnaliser les reponses.",
        "pourquoi": "Elle est utile parce qu'elle evite de tout recommencer a chaque message.",
        "comment": "On enregistre des notes courtes, des sujets et des preferences dans un fichier local.",
        "exemple": "Exemple : retenir que tu prefers le francais et que tu travailles sur l'histoire.",
    },
    "correction": {
        "keywords": ("corrige", "reformule", "orthographe", "grammaire"),
        "definition": "La correction consiste a reparer l'orthographe, les accords et la ponctuation d'un texte.",
        "utilite": "On l'utilise pour rendre une phrase plus claire, plus correcte et plus naturelle.",
        "pourquoi": "Elle est utile parce qu'un texte propre est plus facile a lire et a comprendre.",
        "comment": "On relit la phrase, on corrige les mots fautifs, puis on verifie le style.",
        "exemple": "Exemple : 'je suis aller au college' devient 'Je suis alle au college.'",
    },
    "resume": {
        "keywords": ("resume", "resumer", "synthese", "condense"),
        "definition": "Un resume garde l'idee principale d'un texte en beaucoup moins de mots.",
        "utilite": "On l'utilise pour apprendre plus vite et revoir l'essentiel sans tout relire.",
        "pourquoi": "Il est utile parce qu'il simplifie un contenu long.",
        "comment": "On retire les details secondaires et on garde les points vraiment importants.",
        "exemple": "Exemple : un long cours devient trois ou quatre idees principales.",
    },
    "traduction": {
        "keywords": ("traduire", "traduction", "translate", "langue"),
        "definition": "La traduction consiste a faire passer un texte d'une langue a une autre.",
        "utilite": "On l'utilise pour comprendre un message, apprendre une langue ou communiquer.",
        "pourquoi": "Elle est utile parce qu'elle reduit les barrières de langue.",
        "comment": "On garde le sens, puis on adapte la formulation a la langue cible.",
        "exemple": "Exemple : 'hello' devient 'bonjour'.",
    },
    "quiz": {
        "keywords": ("quiz", "test", "questionnaire", "evaluation"),
        "definition": "Un quiz est une suite de questions qui permet de tester ce qu'on connait.",
        "utilite": "On l'utilise pour s'entrainer et verifier sa memoire.",
        "pourquoi": "Il est utile parce qu'il transforme l'apprentissage en exercice actif.",
        "comment": "On pose une question, on repond, puis on verifie la bonne reponse.",
        "exemple": "Exemple : une question sur Python avec une reponse courte.",
    },
    "code": {
        "keywords": ("code", "fonction", "variable", "boucle", "script"),
        "definition": "Le code est un ensemble d'instructions ecrites pour faire executer une tache a un programme.",
        "utilite": "On l'utilise pour creer des applications, automatiser des actions et traiter des donnees.",
        "pourquoi": "Il est utile parce qu'il donne des instructions precises a la machine.",
        "comment": "On combine des variables, des conditions, des boucles et des fonctions.",
        "exemple": "Exemple : une fonction qui additionne deux nombres.",
    },
    "histoire": {
        "keywords": ("histoire", "historique", "napoleon", "guerre", "revolution"),
        "definition": "L'histoire etudie le passe des peuples, des pays et des evenements.",
        "utilite": "On l'utilise pour comprendre comment les societes ont evolue.",
        "pourquoi": "Elle est utile parce qu'elle aide a comprendre le present.",
        "comment": "On lit des sources, on compare les faits et on construit un recit.",
        "exemple": "Exemple : la Revolution francaise ou le regne de Napoleon.",
    },
    "science": {
        "keywords": ("science", "scientifique", "physique", "chimie", "biologie"),
        "definition": "La science cherche a comprendre le monde avec des observations et des experiences.",
        "utilite": "On l'utilise pour expliquer des phenomenes et creer des technologies.",
        "pourquoi": "Elle est utile parce qu'elle repose sur des tests et des preuves.",
        "comment": "On observe, on formule une hypothese et on verifie par experience.",
        "exemple": "Exemple : etudier la lumiere, la matiere ou le vivant.",
    },
    "espace": {
        "keywords": ("espace", "astronomie", "planete", "orbite", "cosmos"),
        "definition": "L'espace est la region au-dela de l'atmosphere terrestre.",
        "utilite": "On l'utilise pour etudier les planetes, les etoiles et les missions spatiales.",
        "pourquoi": "Il est utile a explorer parce qu'il nous aide a comprendre l'univers.",
        "comment": "Les astronomes observent le ciel et les ingenieurs construisent des missions.",
        "exemple": "Exemple : une sonde qui etudie Mars.",
    },
    "gravite": {
        "keywords": ("gravite", "pesanteur", "attraction"),
        "definition": "La gravite est la force qui attire les corps les uns vers les autres.",
        "utilite": "Elle explique pourquoi les objets tombent et pourquoi les planetes restent en orbite.",
        "pourquoi": "Elle existe parce que toute masse attire les autres masses.",
        "comment": "Plus un objet est massif, plus son attraction peut etre importante.",
        "exemple": "Exemple : la Terre attire la Lune et fait tomber une pomme.",
    },
    "college": {
        "keywords": ("college", "collegien", "scolaire", "ecole"),
        "definition": "Le college est un etablissement scolaire pour les eleves apres l'ecole primaire.",
        "utilite": "On l'utilise pour apprendre les bases dans plusieurs matieres.",
        "pourquoi": "Il est utile parce qu'il prepare a la suite des etudes.",
        "comment": "Les eleves suivent des cours, des devoirs et des evaluations.",
        "exemple": "Exemple : une classe de sixieme ou de quatrieme.",
    },
    "grammaire": {
        "keywords": ("grammaire", "orthographe", "conjugaison", "syntax"),
        "definition": "La grammaire decrit comment construire correctement les phrases.",
        "utilite": "On l'utilise pour ecrire et parler de facon claire et correcte.",
        "pourquoi": "Elle est utile parce qu'elle rend le sens plus precis.",
        "comment": "On regarde les accords, les temps, l'ordre des mots et la ponctuation.",
        "exemple": "Exemple : accorder le verbe avec le sujet.",
    },
    "moliere": {
        "keywords": ("moliere", "theatre", "comedie"),
        "definition": "Moliere est un grand auteur de theatre francais du XVIIe siecle.",
        "utilite": "On l'etudie pour comprendre la comedie, la critique sociale et le style classique.",
        "pourquoi": "Il est connu parce qu'il a marque le theatre francais.",
        "comment": "Ses pieces melangent humour, observation sociale et personnages tres vivants.",
        "exemple": "Exemple : Le Misanthrope ou L'Avare.",
    },
    "chats": {
        "keywords": ("chat", "chats", "felin"),
        "definition": "Le chat est un animal domestique connu pour son agilit et son independance.",
        "utilite": "On l'aime souvent pour sa compagnie et son comportement calme.",
        "pourquoi": "Il est utile dans la vie quotidienne comme animal de compagnie.",
        "comment": "Il observe, se repose beaucoup et communique avec des attitudes et des miaulements.",
        "exemple": "Exemple : un chat qui dort au soleil ou qui joue avec un jouet.",
    },
    "ingenieur": {
        "keywords": ("ingenieur", "ingenieure", "ingénieur"),
        "definition": "Un ingenieur conçoit, construit et ameliore des systemes, des machines ou des logiciels.",
        "utilite": "On l'utilise pour resoudre des problemes concrets avec une methode rigoureuse.",
        "pourquoi": "Ce metier est utile parce qu'il transforme des idees en solutions concretes.",
        "comment": "Il analyse un besoin, propose une solution et la teste.",
        "exemple": "Exemple : un ingenieur logiciel qui fabrique une application.",
    },
    "navigateur": {
        "keywords": ("navigateur", "browser", "chrome", "edge", "firefox"),
        "definition": "Un navigateur est un programme qui affiche des pages web.",
        "utilite": "On l'utilise pour visiter des sites, lire des pages et interagir avec le web.",
        "pourquoi": "Il est utile parce qu'il sert d'interface entre l'utilisateur et le web.",
        "comment": "Il demande une page a un serveur, puis affiche le contenu telecharge.",
        "exemple": "Exemple : Chrome ou Firefox.",
    },
    "api": {
        "keywords": ("api", "endpoint", "request", "reponse"),
        "definition": "Une API est une interface qui permet a deux programmes de communiquer.",
        "utilite": "On l'utilise pour envoyer des requetes et recuperer des donnees.",
        "pourquoi": "Elle est utile parce qu'elle connecte proprement deux systemes.",
        "comment": "Un client envoie une requete, le serveur renvoie une reponse structuree.",
        "exemple": "Exemple : recuperer la liste des messages d'une application.",
    },
    "base de donnees": {
        "keywords": ("base de donnees", "donnees", "table", "sql", "sqlite"),
        "definition": "Une base de donnees sert a stocker et organiser de l'information.",
        "utilite": "On l'utilise pour garder des donnees, les trier et les retrouver vite.",
        "pourquoi": "Elle est utile parce qu'elle centralise l'information de facon fiable.",
        "comment": "On cree des tables, on ajoute des lignes et on interroge les donnees.",
        "exemple": "Exemple : une table de questions et de reponses.",
    },
    "machine learning": {
        "keywords": ("machine learning", "apprentissage automatique", "entrainement"),
        "definition": "Le machine learning apprend des motifs a partir d'exemples.",
        "utilite": "On l'utilise pour predire, classer et reconnaitre des formes ou des textes.",
        "pourquoi": "Il est utile parce qu'il generalise a partir de donnees.",
        "comment": "On donne des exemples, on ajuste un modele, puis on teste sa prediction.",
        "exemple": "Exemple : reconnaitre si un texte est une question ou une demande.",
    },
    "securite": {
        "keywords": ("securite", "mot de passe", "proteger", "risque", "attaque"),
        "definition": "La securite sert a proteger des donnees, des comptes et des systemes.",
        "utilite": "On l'utilise pour limiter les risques et les acces non autorises.",
        "pourquoi": "Elle est utile parce qu'elle reduit les attaques et les erreurs.",
        "comment": "On utilise des mots de passe forts, des acces limites et des controles.",
        "exemple": "Exemple : proteger une API avec une cle privee.",
    },
    "math": {
        "keywords": ("math", "maths", "calcul", "algebre", "geometrie"),
        "definition": "Les mathematiques etudient les nombres, les formes et les relations logiques.",
        "utilite": "On l'utilise pour calculer, modeliser et raisonner avec precision.",
        "pourquoi": "Elles sont utiles parce qu'elles rendent les raisonnements plus rigoureux.",
        "comment": "On pose un probleme, on applique une methode et on verifie le resultat.",
        "exemple": "Exemple : resoudre une equation ou calculer une aire.",
    },
    "physique": {
        "keywords": ("physique", "energie", "mouvement", "force", "optique"),
        "definition": "La physique etudie la matiere, le mouvement, l'energie et les lois naturelles.",
        "utilite": "On l'utilise pour comprendre comment les objets et les phenomenes fonctionnent.",
        "pourquoi": "Elle est utile parce qu'elle explique beaucoup de choses du monde reel.",
        "comment": "On observe, on mesure et on compare les resultats a une theorie.",
        "exemple": "Exemple : la chute d'un objet ou la lumiere qui se propage.",
    },
    "chimie": {
        "keywords": ("chimie", "molécule", "molecule", "reaction", "atome"),
        "definition": "La chimie etudie la composition de la matiere et ses transformations.",
        "utilite": "On l'utilise pour comprendre les reactions et fabriquer de nouveaux materiaux.",
        "pourquoi": "Elle est utile parce qu'elle relie les substances et leurs changements.",
        "comment": "On combine des substances, on observe la reaction et on analyse le resultat.",
        "exemple": "Exemple : une reaction entre deux produits dans un laboratoire.",
    },
    "biologie": {
        "keywords": ("biologie", "vivant", "cellule", "organisme", "dna", "adn"),
        "definition": "La biologie etudie les etres vivants, leurs cellules et leur fonctionnement.",
        "utilite": "On l'utilise pour comprendre le corps, les plantes et les animaux.",
        "pourquoi": "Elle est utile parce qu'elle explique le vivant et ses mecanismes.",
        "comment": "On observe des cellules, des organes ou des ecosystèmes.",
        "exemple": "Exemple : etudier la respiration ou la croissance d'une plante.",
    },
    "geographie": {
        "keywords": ("geographie", "pays", "continent", "ville", "carte"),
        "definition": "La geographie etudie les lieux, les territoires et les relations entre humains et espaces.",
        "utilite": "On l'utilise pour comprendre les regions, les villes et les flux.",
        "pourquoi": "Elle est utile parce qu'elle aide a lire le monde et ses territoires.",
        "comment": "On utilise des cartes, des donnees et des observations de terrain.",
        "exemple": "Exemple : etudier une ville, un pays ou un continent.",
    },
    "algorithme": {
        "keywords": ("algorithme", "etape", "procedure", "tri"),
        "definition": "Un algorithme est une suite d'etapes pour resoudre un probleme.",
        "utilite": "On l'utilise pour organiser une tache et la faire executer par une machine.",
        "pourquoi": "Il est utile parce qu'il rend le raisonnement reproductible.",
        "comment": "On definit les entrees, les etapes et le resultat attendu.",
        "exemple": "Exemple : trier une liste de nombres.",
    },
    "fonction": {
        "keywords": ("fonction", "parametre", "argument", "retour"),
        "definition": "Une fonction est un bloc de code reutilisable qui fait une tache precise.",
        "utilite": "On l'utilise pour eviter de repeter du code et mieux organiser le programme.",
        "pourquoi": "Elle est utile parce qu'elle simplifie la maintenance.",
        "comment": "On lui donne des entrees, elle traite, puis elle renvoie un resultat.",
        "exemple": "Exemple : une fonction qui calcule un total.",
    },
    "classe": {
        "keywords": ("classe", "objet", "instance", "constructeur"),
        "definition": "Une classe est un modele pour creer des objets en programmation.",
        "utilite": "On l'utilise pour regrouper des donnees et des comportements.",
        "pourquoi": "Elle est utile parce qu'elle structure mieux les programmes complexes.",
        "comment": "On decrit des attributs et des methodes, puis on cree des instances.",
        "exemple": "Exemple : une classe Chat avec un nom et une methode miauler.",
    },
    "variable": {
        "keywords": ("variable", "stocke", "valeur", "etat"),
        "definition": "Une variable sert a stocker une valeur dans un programme.",
        "utilite": "On l'utilise pour garder des donnees temporaires ou manipuler des calculs.",
        "pourquoi": "Elle est utile parce qu'elle permet de nommer une information.",
        "comment": "On donne un nom a une valeur, puis on la reutilise dans le code.",
        "exemple": "Exemple : age = 12.",
    },
    "liste": {
        "keywords": ("liste", "tableau", "array", "elements"),
        "definition": "Une liste est une collection ordonnee d'elements.",
        "utilite": "On l'utilise pour stocker plusieurs valeurs dans un seul objet.",
        "pourquoi": "Elle est utile parce qu'elle facilite le parcours et le tri de donnees.",
        "comment": "On ajoute, retire ou parcourt des elements dans l'ordre.",
        "exemple": "Exemple : une liste de prenoms.",
    },
    "dictionnaire": {
        "keywords": ("dictionnaire", "map", "cle", "valeur"),
        "definition": "Un dictionnaire associe des cles a des valeurs.",
        "utilite": "On l'utilise pour retrouver vite une information a partir d'une cle.",
        "pourquoi": "Il est utile parce qu'il donne un acces clair a des donnees nommees.",
        "comment": "On cherche une cle, puis on recupere la valeur associee.",
        "exemple": "Exemple : nom -> Lucie.",
    },
    "json": {
        "keywords": ("json", "objet", "structure", "donnees"),
        "definition": "JSON est un format de texte pour structurer des donnees.",
        "utilite": "On l'utilise pour echanger des donnees entre applications.",
        "pourquoi": "Il est utile parce qu'il est lisible et standardise.",
        "comment": "On ecrit des paires cle-valeur et des tableaux.",
        "exemple": "Exemple : un objet avec un nom et une age.",
    },
    "git": {
        "keywords": ("git", "commit", "branche", "repository"),
        "definition": "Git est un outil pour suivre l'historique du code.",
        "utilite": "On l'utilise pour collaborer, revenir en arriere et garder des versions.",
        "pourquoi": "Il est utile parce qu'il securise le travail sur un projet.",
        "comment": "On enregistre des modifications sous forme de commits.",
        "exemple": "Exemple : creer une branche pour une nouvelle fonctionnalite.",
    },
    "prompt": {
        "keywords": ("prompt", "instruction", "consigne", "requete"),
        "definition": "Un prompt est le texte ou l'instruction qu'on donne a une IA.",
        "utilite": "On l'utilise pour guider la reponse et cadrer la tache.",
        "pourquoi": "Il est utile parce qu'il aide l'IA a mieux comprendre l'intention.",
        "comment": "On precise le role, la tache et le format attendu.",
        "exemple": "Exemple : explique ce texte en trois points.",
    },
    "ocr": {
        "keywords": ("ocr", "texte image", "reconnaissance", "scan"),
        "definition": "L'OCR est une technique qui extrait du texte depuis une image.",
        "utilite": "On l'utilise pour lire des documents scannes ou des captures.",
        "pourquoi": "Elle est utile parce qu'elle transforme l'image en texte exploitable.",
        "comment": "On analyse les formes visibles, puis on reconstruit les caracteres.",
        "exemple": "Exemple : recuperer le texte d'un document photographie.",
    },
    "robot": {
        "keywords": ("robot", "automate", "machine", "capteur"),
        "definition": "Un robot est une machine capable d'agir automatiquement ou semi-automatiquement.",
        "utilite": "On l'utilise pour repeter des taches, explorer ou assister des humains.",
        "pourquoi": "Il est utile parce qu'il automatise des actions physiques ou logiques.",
        "comment": "Il combine des capteurs, un controleur et des actionneurs.",
        "exemple": "Exemple : un bras robotique ou un aspirateur autonome.",
    },
    "climat": {
        "keywords": ("climat", "meteo", "temperature", "rechauffement"),
        "definition": "Le climat correspond aux conditions moyennes d'une region sur une longue periode.",
        "utilite": "On l'utilise pour comprendre les saisons, les regions et les changements globaux.",
        "pourquoi": "Il est utile a etudier parce qu'il influence la vie et les territoires.",
        "comment": "On observe les temperatures, les pluies et leur evolution dans le temps.",
        "exemple": "Exemple : un climat mediterraneen ou oceanique.",
    },
    "reseau": {
        "keywords": ("reseau", "internet", "wifi", "connexion"),
        "definition": "Un reseau relie plusieurs ordinateurs ou appareils pour qu'ils communiquent.",
        "utilite": "On l'utilise pour echanger des donnees et partager des services.",
        "pourquoi": "Il est utile parce qu'il rend la communication plus rapide et plus large.",
        "comment": "Des appareils se connectent via des liens physiques ou sans fil.",
        "exemple": "Exemple : un ordinateur connecte au wifi.",
    },
    "deploiement": {
        "keywords": ("deploiement", "deploy", "production", "mise en ligne"),
        "definition": "Le deploiement consiste a mettre une application en service pour les utilisateurs.",
        "utilite": "On l'utilise pour rendre un projet accessible en vrai.",
        "pourquoi": "Il est utile parce qu'il passe du test a l'usage reel.",
        "comment": "On prepare le code, on configure l'environnement et on publie l'application.",
        "exemple": "Exemple : mettre une API en ligne sur un serveur.",
    },
    "latence": {
        "keywords": ("latence", "delai", "temps de reponse", "retard"),
        "definition": "La latence est le temps entre une demande et la reponse.",
        "utilite": "On l'utilise pour mesurer la rapidite d'un systeme.",
        "pourquoi": "Elle est utile parce qu'elle influence l'experience utilisateur.",
        "comment": "On mesure le temps passe entre l'envoi et la reception.",
        "exemple": "Exemple : le temps d'affichage d'une page web.",
    },
    "cache": {
        "keywords": ("cache", "memoire temporaire", "stockage rapide"),
        "definition": "Un cache garde des donnees temporaires pour les reutiliser rapidement.",
        "utilite": "On l'utilise pour accelerer les lectures frequentes.",
        "pourquoi": "Il est utile parce qu'il evite de recalculer ou recharger trop souvent.",
        "comment": "On enregistre un resultat deja obtenu pour le reutiliser plus tard.",
        "exemple": "Exemple : garder une page deja chargee.",
    },
}


def normalize(text: str) -> str:
    stripped = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    return " ".join(stripped.strip().lower().split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w']+", normalize(text), flags=re.UNICODE)


@dataclass
class ConversationTurn:
    user: str
    assistant: str


@dataclass
class DocumentMemory:
    title: str
    content: str


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EntitySpan:
    start: int
    end: int
    text: str
    label: str


@dataclass
class EntityPrediction:
    text: str
    entities: list[EntitySpan] = field(default_factory=list)


@dataclass
class RelationPrediction:
    head: str
    tail: str
    label: str
    confidence: float


@dataclass
class KnowledgeSnippet:
    content: str
    score: float
    source: str = ""


@dataclass
class DifyKnowledgeClient:
    api_key: str = DIFY_API_KEY
    dataset_id: str = DIFY_DATASET_ID
    api_base: str = DIFY_API_BASE
    _document_cache: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _chunk_cache: list[KnowledgeSnippet] = field(default_factory=list, repr=False)

    def is_ready(self) -> bool:
        return bool(self.api_key and self.dataset_id)

    def retrieve(self, query: str) -> list[KnowledgeSnippet]:
        if not self.is_ready():
            return []

        remote = self._try_remote_retrieval(query)
        if remote:
            return remote

        if not self._chunk_cache:
            self._chunk_cache = self._load_all_chunks()
        if not self._chunk_cache:
            return []

        scored = [
            KnowledgeSnippet(
                content=item.content,
                score=_text_similarity(query, item.content),
                source=item.source,
            )
            for item in self._chunk_cache
        ]
        scored = [item for item in scored if item.score >= 0.18]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:4]

    def _try_remote_retrieval(self, query: str) -> list[KnowledgeSnippet]:
        try:
            payload = self._request_json(
                "POST",
                f"/datasets/{quote(self.dataset_id, safe='')}/retrieve",
                body={"query": query},
            )
        except Exception as exc:
            if "text-embedding" in str(exc).lower() or "invalid_param" in str(exc).lower():
                return []
            return []

        records = payload.get("records", []) if isinstance(payload, dict) else []
        snippets: list[KnowledgeSnippet] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            segment = record.get("segment", {})
            if not isinstance(segment, dict):
                continue
            content = str(segment.get("content", "")).strip()
            if not content:
                continue
            score = float(record.get("score", 0.0) or 0.0)
            document = segment.get("document", {})
            source = ""
            if isinstance(document, dict):
                source = str(document.get("name", "")).strip()
            snippets.append(KnowledgeSnippet(content=content, score=score, source=source))
        snippets.sort(key=lambda item: item.score, reverse=True)
        return snippets[:4]

    def _load_all_chunks(self) -> list[KnowledgeSnippet]:
        chunks: list[KnowledgeSnippet] = []
        for document in self._list_documents():
            document_id = str(document.get("id", "")).strip()
            document_name = str(document.get("name", "")).strip()
            if not document_id:
                continue
            for chunk in self._list_chunks(document_id):
                content = str(chunk.get("content", "")).strip()
                if not content:
                    continue
                chunks.append(
                    KnowledgeSnippet(
                        content=content,
                        score=1.0,
                        source=document_name,
                    )
                )
        return chunks

    def _list_documents(self) -> list[dict[str, Any]]:
        if self._document_cache:
            return self._document_cache
        payload = self._request_json("GET", f"/datasets/{quote(self.dataset_id, safe='')}/documents")
        documents = payload.get("data", []) if isinstance(payload, dict) else []
        self._document_cache = documents if isinstance(documents, list) else []
        return self._document_cache

    def _list_chunks(self, document_id: str) -> list[dict[str, Any]]:
        page = 1
        chunks: list[dict[str, Any]] = []
        while True:
            payload = self._request_json(
                "GET",
                f"/datasets/{quote(self.dataset_id, safe='')}/documents/{quote(document_id, safe='')}/segments?page={page}&limit=100",
            )
            data = payload.get("data", []) if isinstance(payload, dict) else []
            if isinstance(data, list):
                chunks.extend(item for item in data if isinstance(item, dict))
            has_more = bool(payload.get("has_more", False)) if isinstance(payload, dict) else False
            if not has_more:
                break
            page += 1
        return chunks

    def _request_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.api_base}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=20) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
            raise RuntimeError(error_text or str(exc)) from exc
        except URLError as exc:
            raise RuntimeError(str(exc)) from exc

        payload = json.loads(raw) if raw else {}
        return payload if isinstance(payload, dict) else {}


@dataclass
class IntentClassifier:
    label_counts: Counter[str] = field(default_factory=Counter)
    token_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    label_token_totals: Counter[str] = field(default_factory=Counter)
    vocabulary: set[str] = field(default_factory=set)

    @classmethod
    def train(cls, samples: list[dict[str, str]]) -> "IntentClassifier":
        classifier = cls()
        for sample in samples:
            text = normalize(sample.get("text", ""))
            label = normalize(sample.get("label", ""))
            if not text or not label:
                continue
            classifier.label_counts[label] += 1
            for token in tokenize(text):
                classifier.vocabulary.add(token)
                classifier.token_counts[label][token] += 1
                classifier.label_token_totals[label] += 1
        return classifier

    def is_ready(self) -> bool:
        return bool(self.label_counts)

    def predict(self, text: str) -> IntentPrediction | None:
        if not self.is_ready():
            return None

        tokens = tokenize(text)
        if not tokens:
            return None

        vocab_size = max(len(self.vocabulary), 1)
        total_samples = sum(self.label_counts.values())
        scores: dict[str, float] = {}

        for label, label_count in self.label_counts.items():
            score = math.log(label_count / total_samples)
            token_total = self.label_token_totals[label]
            token_counter = self.token_counts[label]
            for token in tokens:
                score += math.log((token_counter[token] + 1) / (token_total + vocab_size))
            scores[label] = score

        best_label = max(scores, key=scores.get)
        confidence = _softmax_confidence(scores, best_label)
        return IntentPrediction(label=best_label, confidence=confidence, scores=scores)


@dataclass
class EntityExtractor:
    phrases_by_label: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def train(cls, samples: list[dict[str, Any]]) -> "EntityExtractor":
        extractor = cls()
        for sample in samples:
            text = str(sample.get("text", "")).strip()
            entities = sample.get("entities", [])
            if not text or not isinstance(entities, list):
                continue
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                label = normalize(str(entity.get("label", "")))
                surface = str(entity.get("text", "")).strip()
                if not label or not surface:
                    continue
                cleaned = _cleanup_surface(surface)
                if cleaned and cleaned not in extractor.phrases_by_label[label]:
                    extractor.phrases_by_label[label].append(cleaned)
        for label in list(extractor.phrases_by_label.keys()):
            extractor.phrases_by_label[label].sort(key=len, reverse=True)
        return extractor

    def is_ready(self) -> bool:
        return any(self.phrases_by_label.values())

    def predict(self, text: str) -> EntityPrediction | None:
        if not self.is_ready():
            return None

        found: list[EntitySpan] = []
        occupied: list[tuple[int, int]] = []
        for label, phrases in self.phrases_by_label.items():
            for phrase in phrases:
                for start, end in _find_phrase_occurrences(text, phrase):
                    if _overlaps(start, end, occupied):
                        continue
                    found.append(EntitySpan(start=start, end=end, text=text[start:end], label=label))
                    occupied.append((start, end))

        if not found:
            return None

        found.sort(key=lambda span: (span.start, span.end))
        return EntityPrediction(text=text, entities=found)


@dataclass
class RelationExtractor:
    signature_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    label_counts: Counter[str] = field(default_factory=Counter)

    @classmethod
    def train(cls, samples: list[dict[str, Any]]) -> "RelationExtractor":
        extractor = cls()
        for sample in samples:
            text = str(sample.get("text", "")).strip()
            relations = sample.get("relations", [])
            if not text or not isinstance(relations, list):
                continue
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                head = _cleanup_surface(str(relation.get("head", "")))
                tail = _cleanup_surface(str(relation.get("tail", "")))
                label = normalize(str(relation.get("label", "")))
                if not head or not tail or not label:
                    continue
                signature = _relation_signature(text, head, tail)
                if not signature:
                    continue
                extractor.label_counts[label] += 1
                extractor.signature_counts[label][signature] += 1
        return extractor

    def is_ready(self) -> bool:
        return bool(self.label_counts)

    def predict(
        self,
        text: str,
        entities: list[EntitySpan] | None = None,
    ) -> list[RelationPrediction]:
        if not self.is_ready():
            return []

        entity_spans = entities or []
        candidates: list[RelationPrediction] = []
        seen_pairs: set[tuple[int, int]] = set()

        for i, head in enumerate(entity_spans):
            for j, tail in enumerate(entity_spans):
                if i == j:
                    continue
                if (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))

                signature = _relation_signature_from_entities(text, head, tail)
                if not signature:
                    continue

                best_label = None
                best_score = 0
                for label, counts in self.signature_counts.items():
                    score = counts[signature]
                    if score > best_score:
                        best_score = score
                        best_label = label

                if best_label and best_score > 0:
                    confidence = best_score / max(self.label_counts[best_label], 1)
                    candidates.append(
                        RelationPrediction(
                            head=head.text,
                            tail=tail.text,
                            label=best_label,
                            confidence=round(min(confidence, 1.0), 3),
                        )
                    )

        return candidates


@dataclass
class LearningBot:
    memory_path: Path
    model: str = DEFAULT_MODEL
    examples: list[dict[str, str]] = field(default_factory=list)
    example_index: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    example_token_index: dict[str, set[int]] = field(default_factory=dict, init=False, repr=False)
    history: list[ConversationTurn] = field(default_factory=list)
    documents: list[DocumentMemory] = field(default_factory=list)
    memory_notes: list[str] = field(default_factory=list)
    memory_sources: dict[str, list[str]] = field(default_factory=dict)
    conversation_summary: str = ""
    preferences: dict[str, str] = field(default_factory=dict)
    subject_memory: dict[str, list[str]] = field(default_factory=dict)
    subject_briefs: dict[str, str] = field(default_factory=dict)
    last_subject: str = ""
    pending_action: str | None = None
    pending_context: str | None = None
    intent_classifier: IntentClassifier = field(default_factory=IntentClassifier)
    entity_extractor: EntityExtractor = field(default_factory=EntityExtractor)
    relation_extractor: RelationExtractor = field(default_factory=RelationExtractor)
    dify_client: DifyKnowledgeClient = field(default_factory=DifyKnowledgeClient)
    startup_warning: str | None = None
    api_available: bool = False

    @classmethod
    def load(cls, memory_path: Path) -> "LearningBot":
        bot = cls(memory_path=memory_path)

        if not memory_path.exists():
            bot.examples = _merge_examples(
                _load_qa_seed(memory_path.parent.parent),
                _load_qa_model(memory_path.parent.parent),
                _load_extra_qa_examples(memory_path.parent.parent),
                [item.copy() for item in DEFAULT_EXAMPLES],
            )
            bot.startup_warning = "Aucune memoire trouvee. La memoire de base a ete chargee."
            bot.intent_classifier = _load_intent_classifier(memory_path.parent.parent)
            bot.entity_extractor = _load_entity_extractor(memory_path.parent.parent)
            bot.relation_extractor = _load_relation_extractor(memory_path.parent.parent)
            bot.dify_client = _load_dify_client()
            bot._rebuild_example_indexes()
            return bot

        try:
            data = json.loads(memory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            bot.examples = _merge_examples(
                _load_qa_seed(memory_path.parent.parent),
                _load_qa_model(memory_path.parent.parent),
                _load_extra_qa_examples(memory_path.parent.parent),
                [item.copy() for item in DEFAULT_EXAMPLES],
            )
            bot.startup_warning = (
                "Le fichier de memoire etait abime. Une memoire de secours a ete chargee."
            )
            bot.intent_classifier = _load_intent_classifier(memory_path.parent.parent)
            bot.entity_extractor = _load_entity_extractor(memory_path.parent.parent)
            bot.relation_extractor = _load_relation_extractor(memory_path.parent.parent)
            bot.dify_client = _load_dify_client()
            bot._rebuild_example_indexes()
            return bot
        except OSError as exc:
            bot.examples = _merge_examples(
                _load_qa_seed(memory_path.parent.parent),
                _load_qa_model(memory_path.parent.parent),
                _load_extra_qa_examples(memory_path.parent.parent),
                [item.copy() for item in DEFAULT_EXAMPLES],
            )
            bot.startup_warning = f"Lecture memoire impossible: {exc}"
            bot.intent_classifier = _load_intent_classifier(memory_path.parent.parent)
            bot.entity_extractor = _load_entity_extractor(memory_path.parent.parent)
            bot.relation_extractor = _load_relation_extractor(memory_path.parent.parent)
            bot.dify_client = _load_dify_client()
            bot._rebuild_example_indexes()
            return bot

        bot.examples = _merge_examples(
            _load_qa_seed(memory_path.parent.parent),
            _load_qa_model(memory_path.parent.parent),
            _load_extra_qa_examples(memory_path.parent.parent),
            [item.copy() for item in DEFAULT_EXAMPLES],
            _load_examples(data.get("examples", [])),
        )
        bot.history = _load_history(data.get("history", []))
        bot.documents = _load_documents(data.get("documents", []))
        bot.memory_notes = _load_memory_notes(data.get("memory_notes", []))
        bot.memory_sources = _load_memory_sources(data.get("memory_sources", {}))
        bot.conversation_summary = str(data.get("conversation_summary", "")).strip()
        bot.preferences = _load_preferences(data.get("preferences", {}))
        bot.subject_memory = _load_subject_memory(data.get("subject_memory", {}))
        bot.subject_briefs = _load_subject_briefs(data.get("subject_briefs", {}))
        bot._rebuild_subject_briefs()
        bot.last_subject = normalize(str(data.get("last_subject", "")))
        bot.model = str(data.get("model", DEFAULT_MODEL))
        bot.api_available = bool(data.get("api_available", False))
        bot.pending_action = None
        bot.pending_context = None
        bot.intent_classifier = _load_intent_classifier(memory_path.parent.parent)
        bot.entity_extractor = _load_entity_extractor(memory_path.parent.parent)
        bot.relation_extractor = _load_relation_extractor(memory_path.parent.parent)
        bot.dify_client = _load_dify_client()
        bot._rebuild_example_indexes()
        return bot

    def _rebuild_example_indexes(self) -> None:
        self.example_index = {}
        token_index: defaultdict[str, set[int]] = defaultdict(set)
        for index, item in enumerate(self.examples):
            question = normalize(item.get("question", ""))
            answer = item.get("answer", "").strip()
            if not question or not answer:
                continue
            item["question"] = question
            item["answer"] = answer
            self.example_index[question] = answer
            for token in set(tokenize(question)):
                if len(token) > 2 and token not in EXAMPLE_INDEX_STOP_WORDS:
                    token_index[token].add(index)
        self.example_token_index = dict(token_index)

    def save(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "api_available": self.api_available,
            "examples": self.examples,
            "documents": [
                {"title": doc.title, "content": doc.content}
                for doc in self.documents[-20:]
            ],
            "memory_notes": self.memory_notes[-20:],
            "memory_sources": self.memory_sources,
            "conversation_summary": self.conversation_summary,
            "preferences": self.preferences,
            "subject_memory": self.subject_memory,
            "subject_briefs": self.subject_briefs,
            "last_subject": self.last_subject,
            "pending_action": self.pending_action,
            "pending_context": self.pending_context,
            "history": [
                {"user": turn.user, "assistant": turn.assistant}
                for turn in self.history[-MAX_HISTORY_TURNS:]
            ],
        }
        self.memory_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def teach(self, question: str, answer: str) -> None:
        question_n = normalize(question)
        answer_n = answer.strip()
        if not question_n or not answer_n:
            raise ValueError("La question et la reponse doivent etre remplies.")

        for item in self.examples:
            if item["question"] == question_n:
                item["answer"] = answer_n
                self.example_index[question_n] = answer_n
                return

        self.examples.append({"question": question_n, "answer": answer_n})
        index = len(self.examples) - 1
        self.example_index[question_n] = answer_n
        for token in set(tokenize(question_n)):
            if len(token) > 2 and token not in EXAMPLE_INDEX_STOP_WORDS:
                self.example_token_index.setdefault(token, set()).add(index)

    def list_knowledge(self) -> list[tuple[str, str]]:
        return [(item["question"], item["answer"]) for item in self.examples]

    def list_documents(self) -> list[tuple[str, str]]:
        return [(doc.title, doc.content) for doc in self.documents]

    def list_memory_notes(self) -> list[str]:
        return list(self.memory_notes)

    def list_memory_sources(self) -> dict[str, list[str]]:
        return {key: list(values) for key, values in self.memory_sources.items()}

    def get_conversation_summary(self) -> str:
        if self.conversation_summary.strip():
            return self.conversation_summary.strip()
        return self._build_conversation_summary()

    def list_preferences(self) -> dict[str, str]:
        return dict(self.preferences)

    def synthetic_example_count(self) -> int:
        return (
            len(SYNTHETIC_TOPIC_FACTS)
            * len(SYNTHETIC_QUESTION_KINDS)
            * len(SYNTHETIC_RESPONSE_MODES)
            * SYNTHETIC_VARIANTS_PER_PAIR
        )

    def total_example_count(self) -> int:
        return len(self.examples) + self.synthetic_example_count()

    def list_subjects(self) -> dict[str, list[str]]:
        return {key: list(values) for key, values in self.subject_memory.items()}

    def list_subject_briefs(self) -> dict[str, str]:
        return dict(self.subject_briefs)

    def predict_intent(self, message: str) -> IntentPrediction | None:
        return self.intent_classifier.predict(message)

    def predict_entities(self, message: str) -> EntityPrediction | None:
        return self.entity_extractor.predict(message)

    def predict_relations(self, message: str) -> list[RelationPrediction]:
        entities = self.predict_entities(message)
        if not entities:
            return []
        return self.relation_extractor.predict(message, entities.entities)

    def set_pending_action(self, action: str, context: str | None = None) -> None:
        self.pending_action = normalize(action) or None
        self.pending_context = context.strip() if context else None
        try:
            self.save()
        except OSError:
            pass

    def clear_pending_action(self) -> None:
        self.pending_action = None
        self.pending_context = None
        try:
            self.save()
        except OSError:
            pass

    def refresh_conversation_summary(self) -> None:
        self.conversation_summary = self._build_conversation_summary()
        try:
            self.save()
        except OSError:
            pass

    def remember_note(self, note: str) -> None:
        self.remember_note_from_source("conversation", note)

    def remember_note_from_source(self, source: str, note: str) -> None:
        note_n = self._normalize_memory_note(note)
        if not note_n:
            return
        if note_n not in self.memory_notes:
            self.memory_notes.append(note_n)
            self.memory_notes = self.memory_notes[-20:]
        source_key = normalize(source) or "other"
        bucket = self.memory_sources.setdefault(source_key, [])
        if note_n not in bucket:
            bucket.append(note_n)
            self.memory_sources[source_key] = bucket[-20:]
        try:
            self.save()
        except OSError:
            pass

    def remember_preference(self, key: str, value: str) -> None:
        key_n = normalize(key)
        value_n = " ".join(value.strip().split())
        if not key_n or not value_n:
            return
        self.preferences[key_n] = value_n
        try:
            self.save()
        except OSError:
            pass

    def remember_subject(self, subject: str, note: str) -> None:
        subject_n = normalize(subject)
        note_n = " ".join(note.strip().split())
        if not subject_n or not note_n:
            return
        bucket = self.subject_memory.setdefault(subject_n, [])
        if note_n not in bucket:
            bucket.append(note_n)
            self.subject_memory[subject_n] = bucket[-10:]
            try:
                self.save()
            except OSError:
                pass

    def add_document(self, title: str, content: str) -> None:
        title_n = " ".join(str(title).strip().split()) or f"Document {len(self.documents) + 1}"
        content_n = " ".join(str(content).strip().split())
        if not content_n:
            raise ValueError("Le document ne peut pas etre vide.")
        self.documents.append(DocumentMemory(title=title_n, content=content_n))
        self.documents = self.documents[-20:]
        self.remember_note_from_source("document", f"Document ajoute: {title_n}")
        try:
            self.save()
        except OSError:
            pass

    def _normalize_memory_note(self, note: str) -> str:
        cleaned = " ".join(str(note).strip().split())
        if not cleaned:
            return ""
        if cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        return cleaned

    def _is_math_request(self, message: str) -> bool:
        message_n = normalize(message)
        if not message_n:
            return False
        if any(phrase in message_n for phrase in MATH_LEAD_INS):
            return True
        expression = self._extract_math_expression(message_n)
        if expression is None:
            return False
        expression_text = expression[0] if isinstance(expression, tuple) else expression
        return bool(re.search(r"[+\-*/%^()]|\b(sqrt|abs|round|pow|ceil|floor)\b", expression_text))

    def _answer_calculation(self, text: str) -> str | None:
        raw = " ".join(text.strip().split())
        if not raw:
            return "Envoie-moi un calcul clair, par exemple 2 + 2 ou 12 * (3 + 4)."

        expression, display = self._extract_math_expression(raw, keep_display=True)
        if not expression:
            if any(phrase in normalize(raw) for phrase in MATH_LEAD_INS):
                return "Je peux calculer, mais j'ai besoin de l'expression exacte. Essaie par exemple 2 + 2 ou 12 * (3 + 4)."
            return None

        try:
            result = self._safe_math_eval(expression)
        except Exception:
            return "Je n'arrive pas ? calculer cette expression. Essaie avec des nombres et des symboles simples."

        formatted_result = self._format_math_number(result)
        explanation = f"{display} = {formatted_result}"
        detail = "Je peux aussi d?tailler le calcul ?tape par ?tape si tu veux."
        return self._format_response("Calcul", [explanation], detail)

    def _extract_math_expression(self, text: str, keep_display: bool = False) -> tuple[str | None, str | None] | str | None:
        lowered = normalize(text)
        if not lowered:
            return (None, None) if keep_display else None

        candidate = lowered.replace("?", "*").replace("?", "/").replace("^", "**")
        candidate = re.sub(
            r"(calcul|calcule|combien font|combien fait|que vaut|resous|resoudre|fais le calcul de|fait le calcul de|donne moi le resultat de|donne-moi le resultat de)\s*",
            "",
            candidate,
        )
        candidate = re.sub(r"(merci|s'il te plait|s il te plait|stp|svp)", "", candidate)
        candidate = re.sub(
            r"(?P<num>\d+(?:[.,]\d+)?)\s*%\s*(?:de|du|des)\s*(?P<other>\d+(?:[.,]\d+)?)",
            r"((\g<num>)/100)*(\g<other>)",
            candidate,
        )
        candidate = re.sub(r"racine\s+de\s+(?P<num>\d+(?:[.,]\d+)?)", r"sqrt(\g<num>)", candidate)
        candidate = re.sub(
            r"(?P<a>\d+(?:[.,]\d+)?)\s+puissance\s+(?:de\s+)?(?P<b>\d+(?:[.,]\d+)?)",
            r"(\g<a>**\g<b>)",
            candidate,
        )
        candidate = re.sub(
            r"(?P<a>\d+(?:[.,]\d+)?)\s+(?:fois|multiplie(?:r)?\s+par|x)\s+(?P<b>\d+(?:[.,]\d+)?)",
            r"(\g<a>*\g<b>)",
            candidate,
        )
        candidate = re.sub(
            r"(?P<a>\d+(?:[.,]\d+)?)\s+(?:divise(?:r)?\s+par|sur)\s+(?P<b>\d+(?:[.,]\d+)?)",
            r"(\g<a>/\g<b>)",
            candidate,
        )
        candidate = re.sub(r"(?P<a>\d+(?:[.,]\d+)?)\s+plus\s+(?P<b>\d+(?:[.,]\d+)?)", r"(\g<a>+\g<b>)", candidate)
        candidate = re.sub(r"(?P<a>\d+(?:[.,]\d+)?)\s+moins\s+(?P<b>\d+(?:[.,]\d+)?)", r"(\g<a>-\g<b>)", candidate)
        candidate = re.sub(r"(?<=\d),(?=\d)", ".", candidate)
        candidate = re.sub(r"(?<=\d)\s*[x]\s*(?=\d)", "*", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()

        tokens = re.findall(r"\*\*|sqrt|abs|round|pow|ceil|floor|\d+(?:\.\d+)?|[+\-*/%^(),]", candidate)
        if not tokens:
            return (None, None) if keep_display else None
        expression = " ".join(tokens)
        if not re.search(r"\d", expression):
            return (None, None) if keep_display else None
        return (expression, expression) if keep_display else expression

    def _safe_math_eval(self, expression: str) -> float:
        node = ast.parse(expression, mode="eval")

        def evaluate(expr: ast.AST) -> float:
            if isinstance(expr, ast.Expression):
                return evaluate(expr.body)
            if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
                return float(expr.value)
            if isinstance(expr, ast.BinOp):
                left = evaluate(expr.left)
                right = evaluate(expr.right)
                if isinstance(expr.op, ast.Add):
                    return left + right
                if isinstance(expr.op, ast.Sub):
                    return left - right
                if isinstance(expr.op, ast.Mult):
                    return left * right
                if isinstance(expr.op, ast.Div):
                    return left / right
                if isinstance(expr.op, ast.FloorDiv):
                    return left // right
                if isinstance(expr.op, ast.Mod):
                    return left % right
                if isinstance(expr.op, ast.Pow):
                    return left ** right
                raise ValueError("Operation non autorisee")
            if isinstance(expr, ast.UnaryOp):
                operand = evaluate(expr.operand)
                if isinstance(expr.op, ast.UAdd):
                    return operand
                if isinstance(expr.op, ast.USub):
                    return -operand
                raise ValueError("Operation non autorisee")
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
                func = MATH_FUNCTIONS.get(expr.func.id)
                if func is None:
                    raise ValueError("Fonction non autorisee")
                args = [evaluate(arg) for arg in expr.args]
                return float(func(*args))
            raise ValueError("Expression non autorisee")

        return evaluate(node)

    def _format_math_number(self, value: float) -> str:
        if math.isfinite(value) and abs(value - round(value)) < 1e-10:
            return str(int(round(value)))
        formatted = f"{value:.12f}".rstrip("0").rstrip(".")
        return formatted or "0"

    def answer(self, message: str) -> str:
        message_n = normalize(message)
        if not message_n:
            return "Ecris une question, ou utilise /teach question | reponse."

        subject = self._resolve_subject_context(message, self._detect_subject(message))

        if message_n.startswith("/doc") or message_n.startswith("/document"):
            title, content = self._parse_document_command(message)
            self.add_document(title, content)
            return f"C'est notÃ©. J'ai ajoutÃ© le document {title}."

        if self.pending_action == "correction":
            target = message.strip()
            self.clear_pending_action()
            return self._answer_correction(target)

        if self.pending_action == "question":
            target = message.strip()
            self.clear_pending_action()
            return self._answer_question(target, subject)

        if self.pending_action == "explain":
            target = message.strip()
            self.clear_pending_action()
            return self._answer_explanation(target)

        if self.pending_action == "translate":
            target = message.strip()
            self.clear_pending_action()
            return self._answer_translation(target)

        if self.pending_action == "plan":
            target = message.strip()
            self.clear_pending_action()
            return self._answer_plan(target)

        if self.pending_action == "quiz":
            return self._answer_quiz(message)

        if self.pending_action == "summary":
            target = message.strip()
            self.clear_pending_action()
            return self._answer_summary(target)

        exact_answer = self._find_exact(message_n)
        if exact_answer is not None:
            self._remember(message, exact_answer)
            return exact_answer

        conversation_control = self._answer_conversation_control(message)
        if conversation_control is not None:
            self._remember(message, conversation_control)
            return conversation_control

        remember_command = self._parse_remember_command(message)
        if remember_command is not None:
            self.remember_note(remember_command)
            return "C'est notÃ©. Je le garderai en mÃ©moire."

        preference_command = self._parse_preference_command(message)
        if preference_command is not None:
            key, value = preference_command
            self.remember_preference(key, value)
            return f"C'est notÃ©. Je retiens ta prÃ©fÃ©rence pour {key}."

        direct_message = self._answer_direct_message(message)
        if direct_message is not None:
            self._remember(message, direct_message)
            return direct_message

        robot_command = self._robot_action_from_message(message)
        if robot_command is not None:
            response = self._send_robot_command(robot_command)
            self._remember_subject("robot", message, response)
            self._remember(message, response)
            return response

        calculation = self._answer_calculation(message)
        if calculation is not None:
            self.remember_subject("math", self._summarize_for_memory(message, calculation))
            self._remember(message, calculation)
            return calculation

        subject = self._resolve_subject_context(message, self._detect_subject(message))

        if self._is_correction_request(message_n):
            self.set_pending_action("correction", message)
            return "D'accord. Envoie-moi la phrase a corriger."

        if self._is_question_request(message_n):
            self.set_pending_action("question", message)
            return "D'accord. Envoie-moi la question."

        if self._is_explain_request(message_n):
            if subject or len(message_n.split()) > 2:
                response = self._answer_explanation(message)
                self._remember(message, response)
                if subject:
                    self._remember_subject(subject, message, response)
                return response
            self.set_pending_action("explain", message)
            return "D'accord. Envoie-moi le texte ou le code a expliquer."

        if self._is_translate_request(message_n):
            self.set_pending_action("translate", message)
            return "D'accord. Envoie-moi le texte a traduire."

        if self._is_plan_request(message_n):
            self.set_pending_action("plan", message)
            return "D'accord. Envoie-moi le texte ou le sujet pour que je fasse un plan."

        if self._is_quiz_request(message_n):
            self.set_pending_action("quiz", "topic")
            return "D'accord. Quel sujet veux-tu pour le quiz ?"

        if self._is_summary_request(message_n):
            self.set_pending_action("summary", message)
            return "D'accord. Envoie-moi le texte a resumer."

        prediction = self.predict_intent(message)
        entities = self.predict_entities(message)
        relations = self.predict_relations(message)
        knowledge = self.predict_knowledge(message)
        document_knowledge = self.predict_documents(message)
        strong_knowledge = bool(knowledge and knowledge[0].score >= 0.25)
        self._capture_memory_from_user(message, subject, prediction, entities, relations)

        if subject and self._is_topic_opening_message(message_n):
            response = self._answer_topic_opening(message, subject)
            self._remember_subject(subject, message, response)
            self._remember(message, response)
            return response

        if subject and self._looks_like_reference_followup(message):
            response = self._answer_followup_on_subject(message, subject)
            self._remember_subject(subject, message, response)
            self._remember(message, response)
            return response

        document_hint = self._answer_from_documents(message)
        if document_hint and self._is_document_query(message_n):
            return document_hint
        local_hint = self._find_exact_or_partial(message_n)
        if local_hint:
            self._remember_subject(subject, message, local_hint)
            self._remember(message, local_hint)
            return local_hint
        memory_hint = self._answer_from_memory(message_n)
        if memory_hint:
            return memory_hint
        subject_hint = self._answer_from_subject_memory(subject, message_n)
        if subject_hint:
            return subject_hint
        if self._is_direct_question(message_n):
            response = self._answer_question(message, subject)
            self._remember(message, response)
            return response
        if strong_knowledge:
            self._remember_subject(subject, message, "knowledge")
            response = self._compose_knowledge_answer(message, entities, relations, knowledge)
            self._remember(message, response)
            return response

        client = self._make_client()
        if client is None:
            answer = self._compose_local_answer(
                message,
                prediction,
                entities,
                relations,
                knowledge,
                document_knowledge,
            )
            self._remember_subject(subject, message, answer)
            self._remember(message, answer)
            return answer

        try:
            answer = self._generate_with_client(client, message, knowledge, document_knowledge)
            if not answer:
                answer = self._compose_local_answer(
                    message,
                    prediction,
                    entities,
                    relations,
                    knowledge,
                    document_knowledge,
                )
                self._remember_subject(subject, message, answer)
                self._remember(message, answer)
                return answer

            self._remember(message, answer)
            self._remember_subject(subject, message, answer)
            self.api_available = True
            return answer
        except Exception as exc:
            self.api_available = False
            self.startup_warning = (
                "Connexion a l'IA distante indisponible. L'application utilise le mode local."
            )
            base = self._compose_local_answer(
                message,
                prediction,
                entities,
                relations,
                knowledge,
                document_knowledge,
            )
            self._remember_subject(subject, message, base)
            self._remember(message, base)
            return f"{base}\n\n[detail: {exc}]"

    def _is_correction_request(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "corrige",
                "reformule",
                "corriger",
                "corrige ca",
                "corrige ?a",
            )
        )

    def _is_question_request(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "question",
                "pose une question",
                "j'ai une question",
                "jai une question",
                "je veux poser une question",
                "aide-moi a repondre",
                "aide moi a repondre",
            )
        )

    def _is_direct_question(self, message_n: str) -> bool:
        if not message_n:
            return False
        if message_n.endswith("?"):
            return True
        return any(
            message_n.startswith(prefix)
            for prefix in (
                "pourquoi",
                "comment",
                "qu'est-ce que",
                "qu est-ce que",
                "qui est",
                "quoi",
            )
        )

    def _is_topic_opening_message(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "parle-moi de",
                "parle moi de",
                "raconte-moi",
                "raconte moi",
                "dis-moi sur",
                "dis moi sur",
                "j'aimerais parler de",
                "jai envie de parler de",
                "je veux parler de",
                "explique-moi sur",
                "explique moi sur",
            )
        )

    def _is_summary_request(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "resume",
                "resumer",
                "fais un resume",
            )
        )

    def _is_robot_request(self, message_n: str) -> bool:
        if not message_n:
            return False
        robot_keywords = (
            "robot",
            "pilote le robot",
            "commande le robot",
            "bouge le robot",
            "deplace le robot",
            "stop robot",
            "arrete le robot",
            "arrête le robot",
        )
        if any(phrase in message_n for phrase in robot_keywords):
            return True
        return message_n in {
            "avance",
            "avancer",
            "recule",
            "reculer",
            "gauche",
            "droite",
            "tourne a gauche",
            "tourne a droite",
            "tourne gauche",
            "tourne droite",
            "stop",
            "arret",
            "arrete",
            "stopper",
        }

    def _robot_duration(self, message: str) -> float:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:s|sec|seconde|secondes)", normalize(message))
        if not match:
            return 0.0
        try:
            return max(0.0, float(match.group(1).replace(",", ".")))
        except ValueError:
            return 0.0

    def _robot_speed(self, message: str, default: int) -> int:
        match = re.search(r"(?:vitesse|speed)\s*(\d{1,3})", normalize(message))
        if match:
            try:
                return max(1, min(100, int(match.group(1))))
            except ValueError:
                pass
        return default

    def _robot_action_from_message(self, message: str) -> dict[str, Any] | None:
        message_n = normalize(message)
        if any(phrase in message_n for phrase in ("test robot", "diagnostic robot", "robot test")):
            return {"action": "diagnostic", "message": "diagnostic"}
        if not self._is_robot_request(message_n):
            return None

        duration = self._robot_duration(message)
        speed = self._robot_speed(message, 45)
        if any(word in message_n for word in ("stop", "arrete", "arrête", "arret", "stopper")):
            return {"action": "stop", "message": "stop"}
        if any(word in message_n for word in ("recule", "reculer", "back", "recul")):
            return {"action": "back", "message": "back", "speed": speed, "duration": duration}
        if any(word in message_n for word in ("gauche", "left")):
            return {"action": "left", "message": "left", "speed": speed, "duration": duration}
        if any(word in message_n for word in ("droite", "right")):
            return {"action": "right", "message": "right", "speed": speed, "duration": duration}
        if any(word in message_n for word in ("avance", "avancer", "forward")):
            return {"action": "forward", "message": "forward", "speed": speed, "duration": duration}
        if "servo" in message_n and re.search(r"\b\d{1,3}\b", message_n):
            angle_match = re.search(r"\b(\d{1,3})\b", message_n)
            angle = int(angle_match.group(1)) if angle_match else 90
            port_match = re.search(r"port\s*(\d+)", message_n)
            slot_match = re.search(r"slot\s*(\d+)", message_n)
            return {
                "action": "servo",
                "message": "servo",
                "port": int(port_match.group(1)) if port_match else 6,
                "slot": int(slot_match.group(1)) if slot_match else 1,
                "angle": max(0, min(180, angle)),
            }
        if "led" in message_n or "rgb" in message_n:
            return {"action": "led", "message": "led", "port": 6, "slot": 1, "index": 0, "r": 0, "g": 120, "b": 255}
        return {"action": "forward", "message": "forward", "speed": speed, "duration": duration}

    def _send_robot_command(self, payload: dict[str, Any]) -> str:
        if not ROBOT_BRIDGE_URL:
            return "Le pont robot n'est pas configure."
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = Request(
                f"{ROBOT_BRIDGE_URL}/",
                data=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8", "ignore") or "{}")
            if data.get("ok") is True:
                action = str(data.get("last_action", payload.get("action", "commande"))).strip()
                friendly = {
                    "forward": "D'accord, j'avance.",
                    "back": "D'accord, je recule.",
                    "left": "D'accord, je tourne à gauche.",
                    "right": "D'accord, je tourne à droite.",
                    "stop": "D'accord, je m'arrête.",
                    "diagnostic": "Je lance le diagnostic.",
                    "servo": "Je règle le servo.",
                    "led": "J'allume les LEDs.",
                }.get(action, f"Robot: {action}.")
                return friendly
            message = str(data.get("message") or data.get("error") or "Robot repondu.")
            return f"Robot: {message}"
        except (URLError, HTTPError, TimeoutError, ValueError, OSError) as exc:
            return f"Robot indisponible: {exc}"

    def robot_bridge_status(self) -> dict[str, Any]:
        if not ROBOT_BRIDGE_URL:
            return {
                "ok": False,
                "message": "Pont robot non configure",
                "connected": False,
                "serial_port": "",
                "board_type": "",
                "last_action": "boot",
                "last_payload": {},
                "last_at": 0.0,
                "last_error": "Pont robot non configure",
                "command_count": 0,
                "uptime_seconds": 0.0,
            }
        try:
            request = Request(f"{ROBOT_BRIDGE_URL}/health", method="GET")
            with urlopen(request, timeout=3) as response:
                data = json.loads(response.read().decode("utf-8", "ignore") or "{}")
            if not isinstance(data, dict):
                raise ValueError("Reponse robot invalide")
            return {
                "ok": bool(data.get("ok", False)),
                "message": str(data.get("message", "")),
                "connected": bool(data.get("connected", False)),
                "serial_port": str(data.get("serial_port", "")),
                "board_type": str(data.get("board_type", "")),
                "last_action": str(data.get("last_action", "boot")),
                "last_payload": data.get("last_payload", {}),
                "last_at": float(data.get("last_at", 0.0) or 0.0),
                "last_error": str(data.get("last_error", "")),
                "command_count": int(data.get("command_count", 0) or 0),
                "uptime_seconds": float(data.get("uptime_seconds", 0.0) or 0.0),
            }
        except (URLError, HTTPError, TimeoutError, ValueError, OSError) as exc:
            return {
                "ok": False,
                "message": f"Robot indisponible: {exc}",
                "connected": False,
                "serial_port": "",
                "board_type": "",
                "last_action": "boot",
                "last_payload": {},
                "last_at": 0.0,
                "last_error": str(exc),
                "command_count": 0,
                "uptime_seconds": 0.0,
            }

    def _is_explain_request(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "explique",
                "expliquer",
                "explication",
                "aide moi a comprendre",
                "aide-moi a comprendre",
            )
        )

    def _is_translate_request(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "traduis",
                "traduire",
                "traduction",
                "translate",
            )
        )

    def _is_plan_request(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "plan",
                "fais un plan",
                "donne un plan",
                "structure",
                "organise",
            )
        )

    def _is_quiz_request(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "quiz",
                "interroge moi",
                "interroge-moi",
                "teste moi",
                "teste-moi",
                "met moi au defi",
                "mets-moi au defi",
                "defi",
            )
        )

    def _parse_remember_command(self, message: str) -> str | None:
        message_n = normalize(message)
        if not message_n:
            return None

        prefixes = (
            "retene que ",
            "souviens-toi que ",
            "mÃ©morise que ",
            "memorise que ",
            "garde en mÃ©moire que ",
            "garde en memoire que ",
        )
        for prefix in prefixes:
            if message_n.startswith(prefix):
                note = message.strip()[len(prefix) :].strip()
                return note or None

        if message_n.startswith("/remember"):
            note = message.strip()[len("/remember") :].strip()
            return note or None

        auto_note = self._extract_memory_fact(message)
        return auto_note

    def _parse_document_command(self, message: str) -> tuple[str, str]:
        payload = ""
        if message.lower().startswith("/document"):
            payload = message[len("/document") :].strip()
        elif message.lower().startswith("/doc"):
            payload = message[len("/doc") :].strip()
        if "|" not in payload:
            raise ValueError("Format attendu: /doc titre | contenu")
        title, content = [part.strip() for part in payload.split("|", 1)]
        if not content:
            raise ValueError("Le contenu du document ne doit pas etre vide.")
        return (title or "Document"), content

    def _parse_preference_command(self, message: str) -> tuple[str, str] | None:
        message_n = normalize(message)
        if not message_n:
            return None

        if message_n.startswith("/pref") and "|" in message:
            payload = message.split("/pref", 1)[-1].strip()
            key, value = [part.strip() for part in payload.split("|", 1)]
            if key and value:
                return key, value

        preference_patterns = [
            (r"(?i)^je prefere\s+(.+)$", "prÃ©fÃ©rence"),
            (r"(?i)^je prÃ©fÃ¨re\s+(.+)$", "prÃ©fÃ©rence"),
            (r"(?i)^j'aime\s+(.+)$", "aime"),
            (r"(?i)^parle plus court$", "style"),
            (r"(?i)^rÃ©ponds plus court$", "style"),
            (r"(?i)^reponds plus court$", "style"),
            (r"(?i)^rÃ©ponds en franÃ§ais$", "langue"),
            (r"(?i)^reponds en francais$", "langue"),
            (r"(?i)^explique simplement$", "niveau"),
        ]
        for pattern, key in preference_patterns:
            match = re.match(pattern, message.strip())
            if not match:
                continue
            if match.groups():
                return key, match.group(1).strip()
            return key, message.strip()
        return None

    def _detect_subject(self, message: str) -> str:
        lowered = normalize(message)
        mapping = {
            "python": ("python", "pyhton", "code"),
            "web": ("web", "html", "css", "site", "internet"),
            "serveur": ("serveur", "backend", "api", "service"),
            "gravite": ("gravite", "gravit?", "pesanteur", "attraction"),
            "navigateur": ("navigateur", "browser"),
            "algorithme": ("algorithme", "algorithm"),
            "json": ("json",),
            "reseau": ("reseau", "r?seau", "wifi"),
            "histoire": ("histoire", "historique", "napoleon", "napolÃ©on", "guerre"),
            "science": ("science", "scientifique", "physique", "chimie", "biologie"),
            "franÃ§ais": ("francais", "franÃ§ais", "grammaire", "orthographe", "conjugaison"),
            "intelligence artificielle": ("ia", "intelligence artificielle", "llm", "modÃ¨le"),
            "college": ("college", "collÃ¨ge", "scolaire", "ecole"),
            "espace": ("espace", "fusÃ©e", "fusÃ©e", "astronomie", "planÃ¨te"),
            "moliere": ("moliere", "moliÃ¨re", "thÃ©Ã¢tre"),
            "chats": ("chat", "chats", "fÃ©lin"),
            "cÃ©line dion": ("celine dion", "cÃ©line dion", "chanteuse"),
        }
        for subject, keywords in mapping.items():
            if any(keyword in lowered for keyword in keywords):
                return subject
        return ""


    def _looks_like_reference_followup(self, message: str) -> bool:
        lowered = normalize(message)
        if not lowered:
            return False
        if self._detect_subject(message):
            return False
        reference_patterns = (
            r"\bson\b",
            r"\bsa\b",
            r"\bses\b",
            r"\bet son\b",
            r"\bet sa\b",
            r"\bet ses\b",
            r"\blui\b",
            r"\belle\b",
            r"\bce personnage\b",
            r"\bcette personne\b",
        )
        return any(re.search(pattern, lowered) for pattern in reference_patterns)

    def _should_use_last_subject(self, message: str) -> bool:
        lowered = normalize(message)
        if not lowered or not self.last_subject:
            return False
        if self._detect_subject(message) or self._extract_explicit_subject_hint(message):
            return False
        if self._looks_like_reference_followup(message):
            return True
        followup_starts = (
            "et ",
            "aussi",
            "continue",
            "continu",
            "la suite",
            "explique plus",
            "plus de detail",
            "plus de details",
            "donne un exemple",
            "exemple",
            "resume",
            "plus simple",
            "simplifie",
        )
        if lowered.startswith(followup_starts):
            return True
        return lowered in {"pourquoi", "comment", "qui", "quand", "ou", "combien"}

    def _answer_direct_message(self, message: str) -> str | None:
        lowered = normalize(message)
        if not lowered:
            return None
        if lowered in {"bonjour", "salut", "coucou", "hey", "hello"}:
            return "Bonjour ! Je suis prete. Pose-moi une question ou donne-moi un sujet."
        if lowered in {"merci", "merci beaucoup", "ok merci"}:
            return "Avec plaisir."
        if lowered in {"ok", "d accord", "daccord", "oui", "non"}:
            return "OK. Dis-moi la suite et je garde le fil."
        if any(phrase in lowered for phrase in ("qui es tu", "qui es-tu", "tu es qui", "comment tu t appelles", "comment tu t'appelles")):
            return (
                "Je suis Lucie, ton assistant local. Je peux repondre, expliquer, corriger, "
                "resumer, calculer et garder le fil d'une discussion."
            )
        if any(phrase in lowered for phrase in ("que peux tu faire", "tu peux faire quoi", "aide moi", "aide-moi")):
            return (
                "Je peux t'aider avec des questions, des calculs, des explications, des resumes, "
                "des corrections et des suivis de conversation. Envoie juste ta demande."
            )
        if any(phrase in lowered for phrase in ("comment ca va", "ca va", "ça va")):
            return "Ca va, je suis prete a travailler avec toi. On avance sur quoi ?"
        return None

    def _extract_explicit_subject_hint(self, message: str) -> str:
        lowered = normalize(message)
        prefixes = (
            "parle-moi de ",
            "parle moi de ",
            "dis-moi de ",
            "dis moi de ",
            "a propos de ",
            "parle de ",
            "dis de ",
            "sur ",
            "concernant ",
        )
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return lowered[len(prefix):].strip(" .!?\t\n\r")
        return ""

    def _resolve_subject_context(self, message: str, subject: str) -> str:
        if subject:
            self.last_subject = subject
            return subject
        hint = self._extract_explicit_subject_hint(message)
        if hint:
            self.last_subject = hint
            return hint
        if self._should_use_last_subject(message):
            return self.last_subject
        return ""

    def _remember_subject(self, subject: str, message: str, answer: str) -> None:
        if not subject:
            return
        note = self._summarize_for_memory(message, answer)
        self.remember_subject(subject, note)
        self.remember_note_from_source("subject", note)
        self._refresh_subject_brief(subject)

    def _summarize_for_memory(self, message: str, answer: str) -> str:
        msg = _strip_memory_markers(message)
        ans = _strip_memory_markers(answer)
        if len(msg) > 90:
            msg = msg[:90].rsplit(" ", 1)[0].rstrip() + "..."
        if len(ans) > 90:
            ans = ans[:90].rsplit(" ", 1)[0].rstrip() + "..."
        return f"Q: {msg} | R: {ans}"

    def _compact_memory_entry(self, note: str) -> str:
        cleaned = _strip_memory_markers(note)
        if " | R: " in cleaned:
            question, answer = cleaned.split(" | R: ", 1)
            question = question.replace("Q: ", "", 1).strip(" .")
            answer = answer.strip(" .")
            if len(question) > 70:
                question = question[:70].rsplit(" ", 1)[0].rstrip() + "..."
            if len(answer) > 140:
                answer = answer[:140].rsplit(" ", 1)[0].rstrip() + "..."
            if question and answer:
                return f"{question} -> {answer}"
            if answer:
                return answer
        return cleaned

    def _refresh_subject_brief(self, subject: str) -> None:
        subject_n = normalize(subject)
        if not subject_n:
            return
        notes = self.subject_memory.get(subject_n, [])
        if not notes:
            self.subject_briefs.pop(subject_n, None)
            return

        pieces: list[str] = [f"Sujet: {subject_n}"]
        recent_notes = [self._compact_memory_entry(note) for note in notes[-3:] if note.strip()]
        if recent_notes:
            short_notes: list[str] = []
            for note in recent_notes[:2]:
                snippet = " ".join(note.strip().split())
                if len(snippet) > 95:
                    snippet = snippet[:95].rsplit(" ", 1)[0].rstrip() + "..."
                short_notes.append(snippet)
            if short_notes:
                pieces.append("A retenir: " + " ; ".join(short_notes))

        related_turn = self._recent_turn_for_subject(subject_n)
        if related_turn:
            user_part = " ".join(related_turn.user.strip().split())
            assistant_part = " ".join(related_turn.assistant.strip().split())
            if len(user_part) > 70:
                user_part = user_part[:70].rsplit(" ", 1)[0].rstrip() + "..."
            if len(assistant_part) > 95:
                assistant_part = assistant_part[:95].rsplit(" ", 1)[0].rstrip() + "..."
            pieces.append(f"Dernier echange: {user_part} -> {assistant_part}")

        brief = " || ".join(pieces)
        if len(brief) > 300:
            brief = brief[:300].rsplit(" ", 1)[0].rstrip() + "..."
        self.subject_briefs[subject_n] = brief

    def _rebuild_subject_briefs(self) -> None:
        rebuilt: dict[str, str] = {}
        for subject in list(self.subject_memory.keys()):
            self._refresh_subject_brief(subject)
            brief = self.subject_briefs.get(normalize(subject), "").strip()
            if brief:
                rebuilt[normalize(subject)] = brief
        self.subject_briefs = rebuilt

    def _expand_followup_subject_query(self, subject: str, message: str) -> str:
        subject_n = normalize(subject)
        message_n = normalize(message)
        if not subject_n or not message_n:
            return subject_n

        followup_templates: list[tuple[tuple[str, ...], str]] = [
            (("mari", "epoux", "epoux", "conjoint"), "mari de {subject}"),
            (("femme", "epouse", "compagne"), "epouse de {subject}"),
            (("enfant", "enfants", "fils", "fille"), "enfants de {subject}"),
            (("carriere", "carrière", "travail", "profession"), "carriere de {subject}"),
            (("naissance", "ne", "nee", "nee", "date de naissance"), "date de naissance de {subject}"),
            (("origine", "nationalite", "nationalite", "pays"), "origine de {subject}"),
            (("age", "ans", "vieux", "vieille"), "age de {subject}"),
            (("vie", "biographie", "parcours"), "biographie de {subject}"),
            (("oeuvre", "oeuvres", "chanson", "album", "livre"), "oeuvre de {subject}"),
        ]
        tokens = set(tokenize(message_n))
        for keywords, template in followup_templates:
                if any(keyword in message_n or keyword in tokens for keyword in keywords):
                    return template.format(subject=subject_n)
        return subject_n

    def _followup_relation_keywords(self, message: str) -> set[str]:
        message_n = normalize(message)
        relation_keywords = {
            "mari",
            "epoux",
            "epouse",
            "conjoint",
            "famille",
            "enfant",
            "enfants",
            "fils",
            "fille",
            "carriere",
            "travail",
            "origine",
            "nationalite",
            "age",
            "naissance",
            "biographie",
            "vie",
            "oeuvre",
            "oeuvres",
            "chanson",
            "album",
        }
        tokens = set(tokenize(message_n))
        return {keyword for keyword in relation_keywords if keyword in message_n or keyword in tokens}

    def _subject_relation_evidence(self, subject: str, message: str) -> str | None:
        subject_n = normalize(subject)
        if not subject_n:
            return None
        keywords = self._followup_relation_keywords(message)
        if not keywords:
            return None

        candidates: list[str] = []
        for note in self.subject_memory.get(subject_n, []):
            note_n = normalize(note)
            if any(keyword in note_n for keyword in keywords):
                candidates.append(self._compact_memory_entry(note))

        if not candidates:
            for turn in reversed(self.history):
                combined = normalize(f"{turn.user} {turn.assistant}")
                if subject_n not in combined:
                    continue
                if any(keyword in combined for keyword in keywords):
                    candidates.append(self._compact_memory_entry(f"Q: {turn.user} | R: {turn.assistant}"))
                    break

        if candidates:
            return candidates[0]
        return None

    def _answer_from_subject_memory(self, subject: str, message_n: str) -> str | None:
        target_subject = subject or self.last_subject
        if not target_subject:
            return None
        brief = self.subject_briefs.get(normalize(target_subject), "").strip()
        notes = self.subject_memory.get(target_subject, [])
        if not notes:
            if brief:
                return brief
            return None
        if any(
            phrase in message_n
            for phrase in (
                f"que sais-tu sur {target_subject}",
                f"que sais tu sur {target_subject}",
                f"qu'est-ce que tu sais sur {target_subject}",
                f"qu est-ce que tu sais sur {target_subject}",
                f"on a parle de {target_subject}",
                f"on a parl? de {target_subject}",
                f"quelles informations as-tu sur {target_subject}",
                f"quelles informations as tu sur {target_subject}",
            )
        ):
            scored_notes = sorted(
                notes,
                key=lambda note: _text_similarity(message_n, note),
                reverse=True,
            )
            recent = [
                self._compact_memory_entry(note)
                for note in scored_notes
                if _text_similarity(message_n, note) >= 0.1
            ][:3]
            if not recent:
                recent = [self._compact_memory_entry(note) for note in notes[-3:]]
            answer = f"On a deja parle de {target_subject} : " + " ; ".join(recent)
            if brief and brief not in answer:
                answer = f"{answer}\n{brief}"
            return answer
        if brief and self._looks_like_reference_followup(message_n):
            return brief
        return None

    def _answer_from_documents(self, message: str) -> str | None:
        if not self.documents:
            return None

        doc = self._find_best_document(message)
        if doc is None:
            return None

        title, content, score = doc
        if score < 0.08:
            return None

        snippet = self._summarize_document_content(content, message)
        return f"D'aprÃ¨s ton document {title} : {snippet}"

    def _is_document_query(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "document",
                "texte",
                "passage",
                "paragraphe",
                "cours",
                "pdf",
                "fichier",
                "note",
            )
        )

    def predict_documents(self, message: str) -> list[KnowledgeSnippet]:
        if not self.documents:
            return []
        snippets: list[KnowledgeSnippet] = []
        for doc in self.documents:
            for chunk in self._split_document_chunks(doc.content):
                score = self._score_document_match(message, doc.title, chunk)
                if score < 0.12:
                    continue
                snippets.append(
                    KnowledgeSnippet(
                        content=chunk,
                        score=score,
                        source=doc.title,
                    )
                )
        snippets.sort(key=lambda item: item.score, reverse=True)
        return snippets[:4]

    def _find_best_document(self, query: str) -> tuple[str, str, float] | None:
        best: tuple[str, str, float] | None = None
        for doc in self.documents:
            for chunk in self._split_document_chunks(doc.content):
                score = self._score_document_match(query, doc.title, chunk)
                candidate = (doc.title, chunk, score)
                if best is None or score > best[2]:
                    best = candidate
        return best

    def _split_document_chunks(self, content: str) -> list[str]:
        chunks: list[str] = []
        for block in re.split(r"\n\s*\n", content):
            part = " ".join(block.strip().split())
            if not part:
                continue
            if len(part) <= 320:
                chunks.append(part)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", part)
            current = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current) + len(sentence) + 1 <= 320:
                    current = f"{current} {sentence}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sentence
            if current:
                chunks.append(current)
        return chunks or [" ".join(content.strip().split())]

    def _summarize_document_content(self, content: str, query: str) -> str:
        chunks = self._split_document_chunks(content)
        best_chunk = max(chunks, key=lambda chunk: _text_similarity(query, chunk))
        snippet = best_chunk.strip()
        if len(snippet) > 280:
            snippet = snippet[:280].rsplit(" ", 1)[0].rstrip() + "..."
        return snippet

    def _score_document_match(self, query: str, title: str, chunk: str) -> float:
        query_n = normalize(query)
        title_n = normalize(title)
        chunk_n = normalize(chunk)
        if not query_n or not chunk_n:
            return 0.0

        chunk_score = _text_similarity(query_n, chunk_n)
        title_score = _text_similarity(query_n, title_n) if title_n else 0.0
        tokens = [token for token in tokenize(query_n) if len(token) > 3]
        bonus = 0.0
        if title_n and tokens and any(token in title_n for token in tokens):
            bonus += 0.06
        if any(
            keyword in query_n
            for keyword in ("rÃ©sume", "resume", "explique", "expliquer", "corrige", "reformule")
        ):
            bonus += 0.03
        return (chunk_score * 0.78) + (title_score * 0.16) + bonus

    def _answer_summary(self, text: str) -> str:
        text = " ".join(text.strip().split())
        if not text:
            return "Envoie-moi le texte Ã  rÃ©sumer."

        client = self._make_client()
        if client is not None:
            answer = self._generate_with_client(
                client,
                text,
                None,
                instructions_override=(
                    "Tu es un assistant de rÃ©sumÃ© en franÃ§ais. "
                    "RÃ©sume le texte de maniÃ¨re claire, concise et fidÃ¨le. "
                    "RÃ©ponds avec le rÃ©sumÃ© seulement."
                ),
                input_override=f"Texte Ã  rÃ©sumer:\n{text}\n\nRÃ©sumÃ©:",
            )
            if answer:
                return answer

        return self._extractive_summary(text)

    def _extractive_summary(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not sentences:
            return text.strip()
        if len(sentences) == 1:
            return sentences[0]
        if len(sentences) == 2:
            return " ".join(sentences)
        summary = " ".join([sentences[0], sentences[1]])
        if len(summary) < len(text) * 0.45 and len(sentences) > 2:
            summary = " ".join([sentences[0], sentences[-1]])
        if len(summary) > 260:
            summary = summary[:260].rsplit(" ", 1)[0].rstrip() + "..."
        return summary

    def _answer_conversation_control(self, message: str) -> str | None:
        message_n = normalize(message)
        if not message_n or not self.history:
            return None
        if self._detect_subject(message) or self._is_direct_question(message_n):
            return None

        last_turn = self.history[-1]
        previous_context = " ".join(
            f"{turn.user} {turn.assistant}" for turn in self.history[-3:]
        ).strip()
        previous_subject = self.last_subject or self._detect_subject(previous_context)

        if any(
            phrase in message_n
            for phrase in (
                "explique plus",
                "developpe",
                "developpe plus",
                "plus de detail",
                "plus de details",
                "detaille",
                "detaille plus",
            )
        ):
            return self._expand_previous_answer(last_turn, previous_subject)

        if any(
            phrase in message_n
            for phrase in (
                "plus simple",
                "explique simplement",
                "version simple",
                "simplifie",
                "je comprend pas",
                "je comprends pas",
            )
        ):
            return self._simplify_previous_answer(last_turn, previous_subject)

        if any(
            phrase in message_n
            for phrase in (
                "donne un exemple",
                "un exemple",
                "exemple concret",
                "par exemple",
            )
        ):
            return self._example_from_previous_answer(last_turn, previous_subject)

        if any(
            phrase in message_n
            for phrase in (
                "resume",
                "resume ca",
                "en bref",
                "plus court",
                "court",
            )
        ):
            return self._summarize_previous_answer(last_turn, previous_subject)

        if message_n in {"continue", "suite", "et apres", "et après", "vas y", "encore"}:
            return self._expand_previous_answer(last_turn, previous_subject)

        return None

    def _previous_topic_label(self, turn: ConversationTurn, subject: str) -> str:
        if subject:
            return subject
        guess = self._detect_subject(f"{turn.user} {turn.assistant}")
        if guess:
            return guess
        text = normalize(turn.user)
        words = [word for word in tokenize(text) if len(word) > 3]
        return words[-1] if words else "ce sujet"

    def _expand_previous_answer(self, turn: ConversationTurn, subject: str) -> str:
        topic = self._previous_topic_label(turn, subject)
        return self._format_response(
            f"Je developpe sur {topic}.",
            [
                f"Tu venais de demander: {turn.user.strip()}",
                f"L'idee de base etait: {turn.assistant.strip()[:220]}",
                "Pour aller plus loin, on regarde la definition, l'utilite, puis un exemple concret.",
            ],
            "Tu peux ensuite me dire: exemple, plus simple, ou compare.",
        )

    def _simplify_previous_answer(self, turn: ConversationTurn, subject: str) -> str:
        topic = self._previous_topic_label(turn, subject)
        compact = self._extractive_summary(turn.assistant)
        if len(compact) > 180:
            compact = compact[:180].rsplit(" ", 1)[0].rstrip() + "..."
        return self._format_response(
            f"Version simple: {topic}",
            [
                compact,
                "En gros: on garde seulement l'idee principale, puis on ajoute un exemple si besoin.",
            ],
            "Dis-moi exemple si tu veux une image plus concrete.",
        )

    def _example_from_previous_answer(self, turn: ConversationTurn, subject: str) -> str:
        topic = self._previous_topic_label(turn, subject)
        examples = {
            "serveur": "Exemple: quand tu ouvres un site, ton navigateur demande la page a un serveur, et le serveur renvoie le contenu.",
            "python": "Exemple: un script Python peut lire un fichier, le trier, puis afficher un resultat automatiquement.",
            "gravite": "Exemple: si tu laches une balle, elle tombe parce que la Terre l'attire vers elle.",
            "intelligence artificielle": "Exemple: une IA peut lire une question, reconnaitre le sujet, puis proposer une reponse utile.",
            "web": "Exemple: une page web utilise HTML pour la structure, CSS pour le style et JavaScript pour les actions.",
        }
        example = examples.get(
            normalize(topic),
            f"Exemple: pour {topic}, prends un cas simple de la vie reelle, observe ce qui entre, ce qui se passe, puis ce qui sort.",
        )
        return self._format_response(
            f"Exemple concret: {topic}",
            [example],
            "Je peux aussi faire un schema en etapes.",
        )

    def _summarize_previous_answer(self, turn: ConversationTurn, subject: str) -> str:
        topic = self._previous_topic_label(turn, subject)
        summary = self._extractive_summary(turn.assistant)
        if len(summary) > 170:
            summary = summary[:170].rsplit(" ", 1)[0].rstrip() + "..."
        return self._format_response(
            f"Resume: {topic}",
            [summary],
            "Je peux aussi le transformer en checklist.",
        )

    def _answer_from_memory(self, message_n: str) -> str | None:
        if not self.memory_notes:
            if not self.preferences and not self.subject_memory and not self.history:
                return None

        if any(
            phrase in message_n
            for phrase in (
                "comment je m'appelle",
                "comment je m appelle",
                "comment je mapelle",
                "qui suis-je",
                "qui suis je",
                "quel est mon nom",
                "mon nom",
                "tu sais ce que je t'ai dit",
                "tu sais ce que je t ai dit",
                "souviens-toi de moi",
                "souviens toi de moi",
                "que sais-tu sur moi",
                "que sais tu sur moi",
            )
        ):
            overview = self._memory_overview()
            if overview:
                return overview

        if any(
            phrase in message_n
            for phrase in (
                "qu'est-ce que j'aime",
                "qu est-ce que j'aime",
                "quels sont mes goÃ»ts",
                "quels sont mes gouts",
                "qu'est-ce que je prefere",
                "qu'est-ce que je prÃ©fÃ¨re",
                "qu est-ce que je prefere",
                "qu est-ce que je prÃ©fÃ¨re",
                "qu'est-ce que tu sais que j'aime",
                "qu est-ce que tu sais que j'aime",
            )
        ):
            for note in reversed(self.memory_notes):
                low = note.lower()
                if low.startswith("L'utilisateur aime".lower()):
                    value = note.split("aime", 1)[-1].strip(" .")
                    if value and value[0].islower():
                        value = value[0].upper() + value[1:]
                    if value:
                        return f"Tu m'as dit aimer {value}."
                if low.startswith("L'utilisateur prefere".lower()):
                    value = note.split("prefere", 1)[-1].strip(" .")
                    if value and value[0].islower():
                        value = value[0].upper() + value[1:]
                    if value:
                        return f"Tu m'as dit prÃ©fÃ©rer {value}."

        if any(
            phrase in message_n
            for phrase in (
                "qu'est-ce que tu sais de moi",
                "qu est-ce que tu sais de moi",
                "que sais-tu de moi",
                "que sais tu de moi",
                "resume ce que tu sais de moi",
                "résume ce que tu sais de moi",
                "de quoi on a parle",
                "de quoi on a parlé",
                "resume notre conversation",
                "résume notre conversation",
            )
        ):
            overview = self.get_conversation_summary()
            if not overview:
                overview = self._memory_overview()
            if overview:
                return overview

        return None

    def _capture_memory_from_user(
        self,
        message: str,
        subject: str,
        prediction: IntentPrediction | None = None,
        entities: EntityPrediction | None = None,
        relations: list[RelationPrediction] | None = None,
    ) -> None:
        fact = self._extract_memory_fact(message)
        if fact:
            self.remember_note_from_source("user", fact)

        cleaned = " ".join(message.strip().split())
        if subject and cleaned and self._looks_like_user_statement(cleaned):
            highlight = self._summarize_user_statement(cleaned, prediction, entities, relations)
            if highlight:
                self.remember_subject(subject, highlight)
                self.remember_note_from_source("conversation", highlight)

    def _looks_like_user_statement(self, text: str) -> bool:
        lowered = normalize(text)
        return any(
            lowered.startswith(prefix)
            for prefix in (
                "je ",
                "j'",
                "j’ai ",
                "j ai ",
                "mon ",
                "ma ",
                "mes ",
                "nous ",
            )
        )

    def _summarize_user_statement(
        self,
        message: str,
        prediction: IntentPrediction | None,
        entities: EntityPrediction | None,
        relations: list[RelationPrediction] | None,
    ) -> str:
        msg = message.strip()
        if len(msg) > 140:
            msg = msg[:140].rsplit(" ", 1)[0].rstrip() + "..."
        summary = f"Q: {msg}"
        if prediction:
            summary += f" | intention={prediction.label}"
        if entities and entities.entities:
            compact = ", ".join(f"{e.text}={e.label}" for e in entities.entities[:2])
            summary += f" | entites={compact}"
        if relations:
            compact = ", ".join(f"{rel.head}->{rel.tail}={rel.label}" for rel in relations[:2])
            summary += f" | relations={compact}"
        return summary

    def _memory_overview(self) -> str | None:
        lines: list[str] = []
        source_counts = {
            source: len(notes)
            for source, notes in sorted(self.memory_sources.items(), key=lambda item: item[0])
            if notes
        }
        if source_counts:
            parts = ", ".join(f"{source}={count}" for source, count in source_counts.items())
            lines.append(f"Ce que je me rappelle de toi :")
            lines.append(f"Sources de souvenirs : {parts}")
        if self.memory_notes:
            lines.extend(f"- {note}" for note in self.memory_notes[-6:])
        if self.preferences:
            pref_lines = ", ".join(f"{key}={value}" for key, value in list(self.preferences.items())[-4:])
            lines.append(f"Preferences : {pref_lines}")
        if self.subject_memory:
            recent_subjects = ", ".join(list(self.subject_memory.keys())[-4:])
            lines.append(f"Sujets suivis : {recent_subjects}")
        if self.subject_briefs:
            brief_subjects = ", ".join(list(self.subject_briefs.keys())[-3:])
            lines.append(f"Fils de sujet : {brief_subjects}")
        if self.last_subject:
            lines.append(f"Sujet courant : {self.last_subject}")
        if self.history:
            recent_turns = [
                f"- Toi: {turn.user[:60]}\n  Moi: {turn.assistant[:90]}"
                for turn in self.history[-4:]
            ]
            lines.append("Derniers echanges :")
            lines.extend(recent_turns)
        return "\n".join(lines) if lines else None

    def _build_conversation_summary(self) -> str:
        parts: list[str] = []
        parts.append("Ce que je me rappelle de notre conversation :")
        if self.memory_notes:
            parts.append(f"Souvenirs directs: {', '.join(self.memory_notes[-4:])}")
        if self.preferences:
            pref_text = ", ".join(f"{key}={value}" for key, value in list(self.preferences.items())[-4:])
            parts.append(f"Preferences: {pref_text}")
        if self.subject_memory:
            subjects = ", ".join(list(self.subject_memory.keys())[-5:])
            parts.append(f"Sujets frequents: {subjects}")
        if self.subject_briefs:
            brief_subjects = ", ".join(list(self.subject_briefs.keys())[-4:])
            parts.append(f"Fils de sujet: {brief_subjects}")
        if self.last_subject:
            parts.append(f"Sujet courant: {self.last_subject}")
        if self.history:
            last_turns = " | ".join(
                f"U: {turn.user[:40]} / A: {turn.assistant[:50]}" for turn in self.history[-3:]
            )
            parts.append(f"Conversations recentes: {last_turns}")
        if self.memory_sources:
            source_text = ", ".join(
                f"{source}={len(notes)}" for source, notes in sorted(self.memory_sources.items())
            )
            parts.append(f"Sources: {source_text}")
        if not parts:
            return "Je n'ai pas encore assez de memoire pour faire un resume."
        summary = " || ".join(parts)
        if len(summary) > 900:
            summary = summary[:900].rsplit(" ", 1)[0].rstrip() + "..."
        return summary

    def _extract_memory_fact(self, message: str) -> str | None:
        text = " ".join(message.strip().split())
        if not text:
            return None

        patterns = [
            (r"(?i)^je\s+m['â€™ ]?appelle\s+(.+)$", "L'utilisateur s'appelle {value}"),
            (r"(?i)^j['â€™]?aime\s+(.+)$", "L'utilisateur aime {value}"),
            (r"(?i)^je\s+prefere\s+(.+)$", "L'utilisateur prefere {value}"),
            (r"(?i)^je\s+prÃ©fÃ¨re\s+(.+)$", "L'utilisateur prefere {value}"),
            (r"(?i)^j'habite\s+(.+)$", "L'utilisateur habite {value}"),
            (r"(?i)^mon sujet prefere est\s+(.+)$", "Le sujet prefere de l'utilisateur est {value}"),
        ]
        for pattern, template in patterns:
            match = re.match(pattern, text)
            if not match:
                continue
            value = match.group(1).strip()
            if not value:
                return None
            return template.format(value=value)
        return None

    def _make_client(self) -> OpenAI | None:
        provider = LLM_PROVIDER
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if provider == "gpt4all":
            base_url = GPT4ALL_BASE_URL or "http://127.0.0.1:4891/v1"
            self.api_available = True
            self.startup_warning = None
            return OpenAI(api_key="local-gpt4all", base_url=base_url)

        if api_key:
            kwargs: dict[str, Any] = {"api_key": api_key}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            self.api_available = True
            self.startup_warning = None
            return OpenAI(**kwargs)

        self.api_available = False
        self.startup_warning = "OPENAI_API_KEY manquante. Le mode local est actif."
        return None

    def _build_instructions(
        self,
        knowledge: list[KnowledgeSnippet] | None = None,
        documents: list[KnowledgeSnippet] | None = None,
    ) -> str:
        facts = "\n".join(
            f"- {item['question']}: {item['answer']}" for item in self.examples[:20]
        )
        memory_notes = "\n".join(f"- {note}" for note in self.memory_notes[-20:])
        current_subject = self.last_subject.strip()
        conversation_summary = self.get_conversation_summary()
        recent_subjects = ", ".join(
            f"{subject}: {', '.join(notes[-2:])}"
            for subject, notes in list(self.subject_memory.items())[-4:]
            if notes
        )
        recent = "\n".join(
            f"Utilisateur: {turn.user}\nAssistant: {turn.assistant}"
            for turn in self.history[-MAX_HISTORY_TURNS:]
        )
        knowledge_block = "\n".join(
            f"- [{snippet.score:.2f}] {snippet.source or 'Dify'}: {snippet.content}"
            for snippet in (knowledge or [])[:4]
        )
        document_block = "\n".join(
            f"- [{snippet.score:.2f}] {snippet.source or 'Document'}: {snippet.content}"
            for snippet in (documents or [])[:4]
        )
        return (
            "Tu es une IA conversationnelle en francais. "
            "Reponds naturellement, avec intelligence et contexte. "
            "Ne poses une question de clarification que si elle est vraiment necessaire. "
            "Si le contexte permet de repondre, repond directement. "
            "Si l'utilisateur ecrit en anglais, tu peux repondre en anglais, mais prefere le francais si ce n'est pas precise.\n\n"
            "Connaissance Dify:\n"
            f"{knowledge_block or '- aucun'}\n\n"
            "Documents utilisateur:\n"
            f"{document_block or '- aucun'}\n\n"
            "Faits appris:\n"
            f"{facts or '- aucun'}\n\n"
            "Memoire longue:\n"
            f"{memory_notes or '- aucune'}\n\n"
            "Resume de conversation:\n"
            f"{conversation_summary or '- aucun'}\n\n"
            "Sujet courant:\n"
            f"{current_subject or '- aucun'}\n\n"
            "Sujets recents:\n"
            f"{recent_subjects or '- aucun'}\n\n"
            "Derniers echanges:\n"
            f"{recent or '- aucun'}\n\n"
            "Regles:\n"
            "- Sois concis quand la question est simple.\n"
            "- Explique quand la demande est complexe.\n"
            "- N'invente pas de faits si tu n'en as pas.\n"
            "- Si tu ne sais pas, dis-le franchement et propose une suite utile.\n"
        )

    def _build_input(
        self,
        message: str,
        knowledge: list[KnowledgeSnippet] | None = None,
        documents: list[KnowledgeSnippet] | None = None,
    ) -> str:
        knowledge_block = "\n".join(
            f"- [{snippet.score:.2f}] {snippet.source or 'Dify'}: {snippet.content}"
            for snippet in (knowledge or [])[:4]
        )
        document_block = "\n".join(
            f"- [{snippet.score:.2f}] {snippet.source or 'Document'}: {snippet.content}"
            for snippet in (documents or [])[:4]
        )
        conversation_summary = self.get_conversation_summary()
        recent_turns = "\n".join(
            f"- U: {turn.user} | A: {turn.assistant}"
            for turn in self.history[-6:]
        )
        return (
            "Contexte conversationnel et demande de l'utilisateur.\n\n"
            f"Resume de conversation:\n{conversation_summary or '- aucun'}\n\n"
            f"Connaissance Dify:\n{knowledge_block or '- aucun'}\n\n"
            f"Documents utilisateur:\n{document_block or '- aucun'}\n\n"
            f"Derniers echanges:\n{recent_turns or '- aucun'}\n\n"
            f"Utilisateur: {message}\n"
            "Assistant:"
        )

    def _generate_with_client(
        self,
        client: OpenAI,
        message: str,
        knowledge: list[KnowledgeSnippet] | None = None,
        documents: list[KnowledgeSnippet] | None = None,
        instructions_override: str | None = None,
        input_override: str | None = None,
    ) -> str:
        instructions = instructions_override or self._build_instructions(knowledge, documents)
        user_input = input_override or self._build_input(message, knowledge, documents)

        try:
            response = client.responses.create(
                model=self.model,
                instructions=instructions,
                input=user_input,
                temperature=0.4,
                max_output_tokens=500,
            )
            answer = self._clean_answer(getattr(response, "output_text", "") or "")
            if answer:
                return answer
        except Exception:
            pass

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.4,
            )
            choice = response.choices[0] if getattr(response, "choices", None) else None
            if choice and getattr(choice, "message", None):
                content = getattr(choice.message, "content", "") or ""
                return self._clean_answer(content)
        except Exception:
            pass

        return ""

    def _answer_correction(self, text: str) -> str:
        client = self._make_client()
        if client is not None:
            answer = self._generate_with_client(
                client,
                text,
                None,
                instructions_override=(
                    "Tu es un correcteur de francais. Corrige la phrase fournie. "
                    "Donne la phrase corrigee, puis une courte explication des changements."
                ),
                input_override=f"Phrase a corriger:\n{text}\n\nReponse attendue:\n- Correction\n- Explication courte",
            )
            if answer:
                return answer

        cleaned = self._apply_french_correction_rules(text)
        if normalize(cleaned) == normalize(text):
            return f"Le texte est deja correct : {cleaned}"

        natural = self._make_more_natural(cleaned)
        changes = self._explain_correction_changes(text, cleaned)
        if natural and normalize(natural) != normalize(cleaned):
            return (
                f"Correction : {cleaned}\n"
                f"Version plus naturelle : {natural}\n"
                f"Explication : {changes}"
            )
        return f"Correction : {cleaned}\nExplication : {changes}"

    def _answer_question(self, text: str, subject: str | None = None) -> str:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return "Envoie-moi une question claire et je te repondrai."

        subject_context = subject
        if not subject_context and self._looks_like_reference_followup(cleaned) and self.last_subject:
            subject_context = self.last_subject
        related_turn = self._best_related_turn(cleaned)
        if related_turn is not None and not (
            self._should_use_last_subject(cleaned) or self._wants_context_answer(normalize(cleaned))
        ):
            related_turn = None

        calculation = self._answer_calculation(cleaned)
        if calculation is not None:
            self._remember(cleaned, calculation)
            self._remember_subject("math", cleaned, calculation)
            return calculation

        learned_answer = self._find_exact_or_partial(normalize(cleaned))
        if learned_answer:
            self._remember(cleaned, learned_answer)
            return learned_answer

        nearest = self._nearest_example(cleaned)
        if nearest:
            best_question, best_answer, score = nearest
            overlap = _token_overlap_score(normalize(cleaned), normalize(best_question))
            if score >= 0.68 and overlap >= 2:
                self._remember(cleaned, best_answer)
                return best_answer

        client = self._make_client()
        if client is not None:
            input_override = (
                f"Resume de conversation:\n{self.get_conversation_summary() or '- aucun'}\n\n"
                f"Question de l'utilisateur:\n{cleaned}\n\nReponse:"
            )
            if subject_context:
                input_override = (
                    f"Resume de conversation:\n{self.get_conversation_summary() or '- aucun'}\n\n"
                    f"Sujet courant: {subject_context}\n"
                    f"Question de l'utilisateur:\n{cleaned}\n\nReponse:"
                )
            if related_turn is not None:
                turn, _ = related_turn
                input_override = (
                    f"Contexte de conversation proche:\n"
                    f"Utilisateur: {turn.user}\n"
                    f"Assistant: {turn.assistant}\n\n"
                    f"{input_override}"
                )
            answer = self._generate_with_client(
                client,
                cleaned,
                instructions_override=(
                    "Tu es une IA conversationnelle en francais. "
                    "Reponds directement, clairement et naturellement. "
                    "Si la question est vague, propose une precision utile. "
                    "Ne mentionne pas de cle API, de serveur, ni de details techniques."
                ),
                input_override=input_override,
            )
            if answer:
                return answer
                return answer

        prediction = self.predict_intent(cleaned)
        entities = self.predict_entities(cleaned)
        relations = self.predict_relations(cleaned)
        knowledge = self.predict_knowledge(cleaned)
        documents = self.predict_documents(cleaned)

        contextual_answer = self._synthesize_context_answer(
            normalize(cleaned),
            prediction,
            entities,
            relations,
            knowledge,
            documents,
        )
        if contextual_answer:
            return contextual_answer

        profile_answer = self._answer_by_question_kind(
            cleaned,
            subject_context,
            related_turn,
            prediction,
            entities,
            relations,
        )
        if profile_answer:
            return profile_answer

        subject_context_answer = self._answer_with_subject_context(
            cleaned,
            subject_context,
            related_turn,
        )
        if subject_context_answer:
            return self._format_response(
                subject_context_answer,
                [],
                self._next_question_hint(subject_context or subject or "", cleaned),
            )

        if related_turn is not None and not self._is_vague_question(cleaned):
            turn, score = related_turn
            if score >= 0.25:
                return self._format_response(
                    "Je reprends le fil de la discussion.",
                    [
                        f"Dernier point proche: {turn.user[:90].strip()}",
                        f"Je retenais aussi: {turn.assistant[:180].strip()}",
                    ],
                    self._next_question_hint(subject_context or subject or "", cleaned),
                )

        common_answer = self._answer_common_question(cleaned)
        if common_answer:
            return self._package_answer(
                "Réponse",
                common_answer,
                detail="Si tu veux, je peux détailler davantage ou donner un exemple.",
                subject=subject_context or "",
                message=cleaned,
            )

        if subject_context and self._looks_like_reference_followup(cleaned):
            return self._package_answer(
                f"Tu parles probablement de {subject_context}.",
                "Dis-moi ce que tu veux savoir exactement sur ce sujet.",
                subject=subject_context,
                message=cleaned,
            )

        cleaned_n = normalize(cleaned)
        if self._is_vague_question(cleaned):
            return self._clarifying_question(cleaned_n)

        nearest = self._nearest_example(cleaned)
        if nearest:
            best_question, best_answer, score = nearest
            if score >= 0.45:
                if score >= 0.8:
                    return best_answer
                return self._package_answer(
                    "Réponse proche",
                    best_answer,
                    detail=f"Je m'appuie sur un exemple proche : {best_question}.",
                    subject=subject_context or "",
                    message=cleaned,
                )

        synthetic_answer = self._answer_from_synthetic_catalog(cleaned)
        if synthetic_answer:
            return synthetic_answer

        if any(word in cleaned_n for word in ("comment", "pourquoi", "qu'est-ce que", "qui est")):
            return self._format_response(
                "Je peux r?pondre, mais j'ai besoin d'un peu plus de contexte.",
                ["Donne-moi le sujet exact ou colle le texte."],
                self._next_question_hint(subject_context or "", cleaned),
            )
        return self._compose_prediction_answer(prediction, entities, relations) if prediction else (
            "Je peux r?pondre, mais je veux ?tre s?r de bien viser. Peux-tu pr?ciser ta question ?"
        )

    def _answer_from_synthetic_catalog(self, message: str) -> str | None:
        message_n = normalize(message)
        topic_key = self._detect_synthetic_topic(message_n)
        if not topic_key:
            return None

        topic = SYNTHETIC_TOPIC_FACTS.get(topic_key)
        if not isinstance(topic, dict):
            return None

        title = str(topic.get("title", topic_key)).strip() or topic_key
        definition = str(topic.get("definition", "")).strip()
        utility = str(topic.get("utilite", "")).strip()
        why = str(topic.get("pourquoi", "")).strip()
        how = str(topic.get("comment", "")).strip()
        example = str(topic.get("exemple", "")).strip()
        mode = self._synthetic_question_kind(message_n)
        headline = f"Réponse: {title}"

        if mode == "definition":
            return self._format_response(
                headline,
                [definition, utility],
                "Si tu veux, je peux aussi donner un exemple concret.",
            )
        if mode == "utilite":
            return self._format_response(
                headline,
                [f"Utilite de {title}", utility, example],
                "Je peux aussi l'expliquer plus simplement si tu veux.",
            )
        if mode == "pourquoi":
            return self._format_response(
                headline,
                [f"Pourquoi {title}", why, utility],
                "Je peux aussi te donner un exemple ou un schema rapide.",
            )
        if mode == "comment":
            return self._format_response(
                headline,
                [f"Comment comprendre {title}", how, example],
                "Je peux aussi te faire un pas-a-pas plus court.",
            )
        if mode == "exemple":
            return self._format_response(
                headline,
                [f"Exemple de {title}", example, definition],
                "Si tu veux, je peux en donner un autre.",
            )
        if mode == "avantage":
            return self._format_response(
                headline,
                [f"Ce qui est utile avec {title}", utility, why],
                "Je peux aussi comparer avec un autre sujet.",
            )
        if mode == "risque":
            return self._format_response(
                headline,
                [f"Point a surveiller pour {title}", "Attention a bien distinguer le sujet et son usage reel.", example],
                "Je peux aussi te donner les erreurs courantes.",
            )
        if mode == "comparaison":
            return self._format_response(
                headline,
                [f"Comparer {title}", "On compare ce sujet avec son usage, ses limites et ses exemples.", utility],
                "Je peux aussi le comparer a un autre concept si tu le veux.",
            )
        if mode == "relation":
            return self._format_response(
                headline,
                [f"Relation autour de {title}", "Le sujet est relie a des cas pratiques, des exemples et des usages.", example],
                "Je peux aussi te montrer les liens avec d'autres notions.",
            )
        if mode == "resume":
            return self._format_response(
                headline,
                [f"Resume de {title}", definition, utility, example],
                "Je peux te le reformuler encore plus court.",
            )
        if mode == "court":
            return self._format_response(
                headline,
                [definition],
                "Je peux aussi faire plus simple ou plus detaille.",
            )
        if mode == "detaille":
            return self._format_response(
                headline,
                [definition, utility, why, how, example],
                "Je peux aussi condenser tout ça en trois lignes.",
            )
        if mode == "explication":
            return self._format_response(
                headline,
                [how, utility, example],
                "Je peux aussi te le faire en version plus simple.",
            )
        if mode == "fonctionnement":
            return self._format_response(
                headline,
                [how, example, utility],
                "Je peux aussi te montrer le schéma logique.",
            )
        if mode == "limite":
            return self._format_response(
                headline,
                [f"Limite principale: {why}", "Il faut aussi regarder le contexte d'utilisation."],
                "Je peux aussi comparer avec un autre concept.",
            )
        if mode == "etapes":
            return self._format_response(
                headline,
                [how, "On peut couper le probleme en petites etapes."],
                "Je peux aussi te le presenter sous forme de checklist.",
            )
        if mode == "usage":
            return self._format_response(
                headline,
                [utility, example],
                "Je peux aussi te donner un cas concret ou un autre exemple.",
            )
        if mode == "conseil":
            return self._format_response(
                headline,
                [f"Conseil : bien comprendre {title} avant de le utiliser en vrai.", utility],
                "Je peux aussi te dire par quoi commencer.",
            )
        if mode == "erreur":
            return self._format_response(
                headline,
                [f"Erreur courante : confondre {title} avec autre chose.", "Mieux vaut partir d'un exemple simple."],
                "Je peux aussi lister les pièges fréquents.",
            )
        if mode == "situation":
            return self._format_response(
                headline,
                [f"En situation, {title} sert surtout dans des cas concrets.", example, utility],
                "Je peux aussi te donner une version scolaire ou pratique.",
            )

        return self._format_response(
            headline,
            [definition, utility, example],
            "Si tu veux, je peux te donner une version plus courte ou plus detaillee.",
        )

    def _synthetic_question_kind(self, message_n: str) -> str:
        if any(word in message_n for word in ("qu'est-ce que", "qu est-ce que", "c'est quoi", "c est quoi", "definis", "definit", "definition")):
            return "definition"
        if any(word in message_n for word in ("explique", "explication", "explique-moi", "explique moi", "dis-moi ce que c'est", "dis moi ce que c'est")):
            return "explication"
        if any(word in message_n for word in ("a quoi sert", "à quoi sert", "sert a", "sert à", "utilite", "utilité")):
            return "utilite"
        if any(word in message_n for word in ("pourquoi", "pour quoi")):
            return "pourquoi"
        if any(word in message_n for word in ("comment", "comment faire", "comment ca marche", "comment ça marche", "comment fonctionne", "fonctionnement")):
            return "comment"
        if any(word in message_n for word in ("exemple", "par exemple", "un cas", "un cas concret")):
            return "exemple"
        if any(word in message_n for word in ("avantage", "interet", "intérêt", "utile", "bénéfice", "benefice")):
            return "avantage"
        if any(word in message_n for word in ("risque", "attention", "piege", "piège", "limite", "limites", "erreur", "erreurs")):
            return "risque"
        if any(word in message_n for word in ("comparer", "compare", "difference", "différence", "diff entre", "différence entre", "différence avec")):
            return "comparaison"
        if any(word in message_n for word in ("relation", "lien", "relie", "lié", "lie", "associe", "associé")):
            return "relation"
        if any(word in message_n for word in ("resume", "résume", "resumer", "résumer", "court", "bref", "brève", "breve")):
            return "resume"
        if any(word in message_n for word in ("detaille", "détaillé", "detaille", "plus de détails", "plus de detail")):
            return "detaille"
        if any(word in message_n for word in ("etapes", "étapes", "pas a pas", "pas à pas", "checklist")):
            return "etapes"
        if any(word in message_n for word in ("usage", "utilisation", "utiliser", "sert a quoi", "sert à quoi")):
            return "usage"
        if any(word in message_n for word in ("conseil", "recommande", "recommandes", "par quoi commencer")):
            return "conseil"
        if any(word in message_n for word in ("situation", "en vrai", "dans la vie", "dans le concret")):
            return "situation"
        return "definition"

    def _detect_synthetic_topic(self, message_n: str) -> str | None:
        best_topic: str | None = None
        best_score = 0
        tokens = set(tokenize(message_n))
        for topic, data in SYNTHETIC_TOPIC_FACTS.items():
            keywords = data.get("keywords", ())
            if not isinstance(keywords, tuple):
                continue
            score = 0
            explicit_hits = 0
            for keyword in keywords:
                keyword_n = normalize(str(keyword))
                if not keyword_n:
                    continue
                if " " in keyword_n:
                    if keyword_n in message_n:
                        explicit_hits += 2
                        score += 2
                    continue
                if keyword_n in tokens:
                    explicit_hits += 1
                    score += 1
                elif keyword_n in message_n:
                    score += 1
            if explicit_hits == 0 and score < 2:
                continue
            if score > best_score:
                best_score = score
                best_topic = topic
        return best_topic

    def _answer_explanation(self, text: str) -> str:
        raw = text.strip()
        cleaned = " ".join(raw.split())
        if not cleaned:
            return "Envoie-moi le texte ou le code a expliquer."

        subject = self._detect_subject(cleaned)
        synthetic_answer = self._answer_from_synthetic_catalog(cleaned)
        if synthetic_answer:
            return synthetic_answer
        if subject:
            subject_answer = self._answer_from_synthetic_catalog(f"explication {subject}")
            if subject_answer:
                return subject_answer

        client = self._make_client()
        if client is not None:
            answer = self._generate_with_client(
                client,
                cleaned,
                instructions_override=(
                    "Tu es un professeur patient en francais. "
                    "Explique le texte ou le code simplement, en t'adressant a un debutant. "
                    "Si c'est du code, fais une explication ligne par ligne ou bloc par bloc. "
                    "Si c'est un texte, resume d'abord l'idee generale puis detaille les points importants."
                ),
                input_override=f"Texte ou code a expliquer:\n{raw}\n\nExplication:",
            )
            if answer:
                return answer

        if self._looks_like_code(raw):
            return self._explain_code_locally(raw)
        return f"En clair : {self._extractive_summary(raw)}"

    def _answer_translation(self, text: str) -> str:
        raw = text.strip()
        if not raw:
            return "Envoie-moi le texte a traduire."

        client = self._make_client()
        if client is not None:
            answer = self._generate_with_client(
                client,
                raw,
                instructions_override=(
                    "Tu es un traducteur expert en francais. "
                    "Traduis le texte de maniere naturelle, sans expliquer inutilement. "
                    "Si la langue source est incertaine, garde le sens le plus probable."
                ),
                input_override=f"Texte a traduire:\n{raw}\n\nTraduction:",
            )
            if answer:
                return answer

        return self._translate_local(raw)

    def _answer_plan(self, text: str) -> str:
        raw = text.strip()
        if not raw:
            return "Envoie-moi un texte ou un sujet pour que je fasse un plan."

        client = self._make_client()
        if client is not None:
            answer = self._generate_with_client(
                client,
                raw,
                instructions_override=(
                    "Tu es un professeur et un organisateur d'idees. "
                    "Transforme le texte en plan clair et hierarchise. "
                    "Reponds avec des titres courts et des puces utiles."
                ),
                input_override=f"Texte ou sujet a organiser:\n{raw}\n\nPlan:",
            )
            if answer:
                return answer

        return self._build_plan_locally(raw)

    def _answer_quiz(self, text: str) -> str:
        raw = " ".join(text.strip().split())
        if not raw:
            return "Quel sujet veux-tu pour le quiz ?"

        context = self._load_quiz_context()
        stage = context.get("stage", "topic")
        if stage == "answer":
            expected_answer = str(context.get("answer", "")).strip()
            question = str(context.get("question", "")).strip()
            self.clear_pending_action()
            if not expected_answer:
                return "Je n'ai plus la bonne reponse en memoire. On recommence ?"
            if self._is_close_answer(raw, expected_answer):
                return f"Bonne reponse. Pour la question « {question} », c'etait bien: {expected_answer}"
            return f"Pas tout a fait. Pour « {question} », la bonne reponse etait: {expected_answer}"

        subject = raw
        question, answer = self._pick_quiz_item(subject)
        self.pending_context = json.dumps(
            {
                "stage": "answer",
                "subject": subject,
                "question": question,
                "answer": answer,
            },
            ensure_ascii=False,
        )
        try:
            self.save()
        except OSError:
            pass
        return f"Question pour {subject}:\n{question}\nReponds en une phrase."

    def _load_quiz_context(self) -> dict[str, str]:
        if not self.pending_context:
            return {"stage": "topic"}
        try:
            payload = json.loads(self.pending_context)
        except json.JSONDecodeError:
            return {"stage": "topic"}
        if isinstance(payload, dict):
            return {str(key): str(value) for key, value in payload.items()}
        return {"stage": "topic"}

    def _pick_quiz_item(self, subject: str) -> tuple[str, str]:
        subject_n = normalize(subject)
        candidates: list[dict[str, str]] = []
        keywords = [token for token in tokenize(subject_n) if len(token) > 2]
        for item in self.examples:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            if not question or not answer:
                continue
            haystack = normalize(f"{question} {answer}")
            if subject_n and subject_n in haystack:
                candidates.append({"question": question, "answer": answer})
                continue
            if keywords and any(keyword in haystack for keyword in keywords):
                candidates.append({"question": question, "answer": answer})
        if not candidates:
            candidates = [item for item in self.examples if item.get("question") and item.get("answer")]
        if not candidates:
            return ("Qu'est-ce que l'IA ?", "Un programme qui apprend a reconnaitre des motifs.")
        index = sum(ord(char) for char in subject_n) % len(candidates) if candidates else 0
        chosen = candidates[index]
        return chosen["question"], chosen["answer"]

    def _is_close_answer(self, answer: str, expected: str) -> bool:
        answer_n = normalize(answer)
        expected_n = normalize(expected)
        if not answer_n or not expected_n:
            return False
        if answer_n == expected_n:
            return True
        if _text_similarity(answer_n, expected_n) >= 0.62:
            return True
        expected_tokens = [token for token in tokenize(expected_n) if len(token) > 3]
        if not expected_tokens:
            return answer_n in expected_n or expected_n in answer_n
        hits = sum(1 for token in expected_tokens if token in answer_n)
        return hits / max(len(expected_tokens), 1) >= 0.5

    def _looks_like_code(self, text: str) -> bool:
        lowered = text.lower()
        if "\n" in text:
            return True
        if any(token in lowered for token in ("def ", "class ", "import ", "return ", "print(", "if ", "for ", "while ")):
            return True
        if any(symbol in text for symbol in ("{", "}", "(", ")", ";", "=>")):
            return True
        return False

    def _translate_local(self, text: str) -> str:
        lower = normalize(text)
        if not lower:
            return "Je n'ai rien a traduire."
        if any(word in lower for word in ("bonjour", "merci", "comment ca va", "ça va", "ca va")):
            return "Traduction simple: Hello / Thank you / How are you?"
        if lower.startswith("hello"):
            return "Traduction simple: Bonjour."
        if lower.startswith("thank you"):
            return "Traduction simple: Merci."
        if lower.startswith("how are you"):
            return "Traduction simple: Comment vas-tu ?"
        return (
            "Je peux traduire ce texte, mais sans modele distant je prefere que tu "
            "me precisions la langue source et la langue cible."
        )

    def _build_plan_locally(self, text: str) -> str:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return "Je n'ai rien a organiser."
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]
        if not sentences:
            sentences = [cleaned]
        pieces = sentences[:5]
        lines = ["Plan rapide:"]
        for index, sentence in enumerate(pieces, start=1):
            if len(sentence) > 120:
                sentence = sentence[:120].rsplit(" ", 1)[0].rstrip() + "..."
            lines.append(f"{index}. {sentence}")
        if len(sentences) > 5:
            lines.append(f"... et {len(sentences) - 5} autre(s) point(s) a developper.")
        return "\n".join(lines)

    def _explain_code_locally(self, text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "Envoie-moi un bout de code a expliquer."

        explanations: list[str] = []
        for line in lines[:8]:
            stripped = line.strip()
            lowered = stripped.lower()
            if stripped.startswith("#"):
                explanations.append(f"- Commentaire: {stripped.lstrip('#').strip()}")
            elif lowered.startswith("def "):
                name = stripped[4:].split("(", 1)[0].strip() or "une fonction"
                explanations.append(f"- Cette ligne definit la fonction {name}.")
            elif lowered.startswith("class "):
                name = stripped[6:].split("(", 1)[0].split(":", 1)[0].strip() or "une classe"
                explanations.append(f"- Cette ligne definit la classe {name}.")
            elif lowered.startswith("import ") or lowered.startswith("from "):
                explanations.append("- Cette ligne importe un module ou une partie de module.")
            elif lowered.startswith("if "):
                explanations.append("- Cette ligne teste une condition.")
            elif lowered.startswith("elif "):
                explanations.append("- Cette ligne teste une autre condition.")
            elif lowered.startswith("else:"):
                explanations.append("- Cette ligne prend le cas contraire.")
            elif lowered.startswith("for "):
                explanations.append("- Cette ligne lance une boucle.")
            elif lowered.startswith("while "):
                explanations.append("- Cette ligne repete une action tant qu'une condition est vraie.")
            elif lowered.startswith("return "):
                explanations.append("- Cette ligne renvoie une valeur.")
            elif "=" in stripped:
                explanations.append("- Cette ligne stocke ou calcule une valeur dans une variable.")
            else:
                explanations.append(f"- Cette ligne fait une etape du programme: {stripped}")

        if len(lines) > 8:
            explanations.append(f"- Et il y a encore {len(lines) - 8} autre(s) ligne(s) a detailler si tu veux.")

        return "Voici une lecture simple du code:\n" + "\n".join(explanations)

    def _answer_common_question(self, text: str) -> str | None:
        lower = normalize(text)
        if not lower:
            return None
        if "pourquoi" in lower and "ciel" in lower and "bleu" in lower:
            return (
                "Le ciel para?t bleu parce que l'atmosph?re diffuse davantage les petites "
                "longueurs d'onde de la lumi?re solaire, comme le bleu."
            )
        if any(word in lower for word in ("qu'est-ce que l'ia", "c'est quoi l'ia", "intelligence artificielle")):
            return (
                "Une intelligence artificielle est un programme qui apprend ? reconna?tre des "
                "motifs et ? produire des r?ponses utiles ? partir de donn?es."
            )
        if any(word in lower for word in ("qu'est-ce que python", "c'est quoi python", "python")):
            return (
                "Python est un langage de programmation simple ? lire et tr?s utilis? pour "
                "l'automatisation, les scripts et l'intelligence artificielle."
            )
        if any(word in lower for word in ("qui est moliere", "qui est moli?re", "moliere", "moli?re")):
            return (
                "Moli?re est un grand auteur de th??tre fran?ais du XVIIe si?cle, connu pour "
                "ses com?dies."
            )
        if "exemple" in lower and "compar" in lower:
            return (
                "Exemple de comparaison: Python est plus simple pour debuter en programmation, "
                "alors que JavaScript est surtout utilise pour rendre les pages web interactives. "
                "L'usage n'est donc pas le meme selon le cas reel."
            )
        return None

    def _question_kind(self, message: str) -> str:
        message_n = normalize(message)
        if not message_n:
            return "unknown"
        if self._looks_like_reference_followup(message):
            return "followup"
        if any(word in message_n for word in ("qu'est-ce que", "qu est-ce que", "c'est quoi", "c est quoi", "definit", "definir", "definition")):
            return "definition"
        if any(word in message_n for word in ("pourquoi", "pour quoi")):
            return "why"
        if any(word in message_n for word in ("comment", "comment faire", "comment ca marche", "comment ça marche", "fonctionnement")):
            return "how"
        if any(word in message_n for word in ("exemple", "par exemple", "un cas", "cas concret")):
            return "example"
        if any(word in message_n for word in ("comparer", "compare", "difference", "différence", "diff entre", "différence entre", "différence avec")):
            return "compare"
        if any(word in message_n for word in ("liste", "quels sont", "donne-moi les", "donne moi les", "recense", "énumère", "enumere")):
            return "list"
        if any(word in message_n for word in ("qui est", "qui sont", "c'est qui", "c est qui")):
            return "who"
        if any(word in message_n for word in ("quand", "date", "moment", "période", "periode")):
            return "when"
        if any(word in message_n for word in ("où", "ou ", "ou se", "ou est", "localisation", "lieu")):
            return "where"
        if any(word in message_n for word in ("combien", "nombre", "compte", "quantité", "quantite")):
            return "count"
        return "general"

    def _recent_turn_for_subject(self, subject: str) -> ConversationTurn | None:
        subject_n = normalize(subject)
        if not subject_n:
            return None
        for turn in reversed(self.history):
            combined = normalize(f"{turn.user} {turn.assistant}")
            if subject_n in combined:
                return turn
        return None

    def _best_related_turn(self, message: str) -> tuple[ConversationTurn, float] | None:
        message_n = normalize(message)
        if not message_n or not self.history:
            return None
        message_tokens = {token for token in tokenize(message_n) if len(token) > 3}

        best: tuple[ConversationTurn, float] | None = None
        for turn in reversed(self.history):
            turn_text = normalize(f"{turn.user} {turn.assistant}")
            turn_tokens = {token for token in tokenize(turn_text) if len(token) > 3}
            shared = bool(message_tokens and turn_tokens and message_tokens & turn_tokens)
            score = max(_text_similarity(message_n, turn.user), _text_similarity(message_n, turn.assistant))
            if not shared and score < 0.42:
                continue
            if best is None or score > best[1]:
                best = (turn, score)
        if best is None or best[1] < 0.25:
            return None
        return best

    def _best_subject_example(self, subject: str) -> tuple[str, str, float] | None:
        subject_n = normalize(subject)
        if not subject_n:
            return None
        subject_tokens = {token for token in tokenize(subject_n) if len(token) > 2}

        prompts = [
            f"parle-moi de {subject_n}",
            f"parle moi de {subject_n}",
            f"qui est {subject_n}",
            f"qu'est-ce que {subject_n}",
            f"qu est-ce que {subject_n}",
            f"explique {subject_n}",
            subject_n,
        ]
        candidates: list[tuple[str, str, float]] = []
        for prompt in prompts:
            nearest = self._nearest_example(prompt)
            if nearest is not None:
                question_n = normalize(nearest[0])
                if subject_tokens and not any(token in question_n for token in subject_tokens) and subject_n not in question_n:
                    continue
                candidates.append(nearest)
        if not candidates:
            return None
        best = max(candidates, key=lambda item: item[2])
        if best[2] < 0.32:
            return None
        return best

    def _answer_topic_opening(self, message: str, subject: str) -> str:
        subject_n = normalize(subject)
        if not subject_n:
            return "Je peux parler de ce sujet. Dis-moi ce que tu veux savoir exactement."

        subject_example = self._best_subject_example(subject_n)
        if subject_example is not None:
            best_question, best_answer, score = subject_example
            if score >= 0.35:
                return self._format_response(
                    f"On peut parler de {subject}.",
                    [
                        f"Je me base sur un exemple proche : {best_question}.",
                        best_answer,
                    ],
                    "Tu peux maintenant poser une question plus precise sur ce sujet et je garde le fil.",
                )

        notes = self.subject_memory.get(subject_n, [])
        if notes:
            recent = [
                self._compact_memory_entry(note)
                for note in notes[-3:]
                if note.strip()
            ]
            if recent:
                return self._format_response(
                    f"On peut parler de {subject}.",
                    [f"Je me rappelle deja: {recent[0]}"]
                    + [f"Autre point: {item}" for item in recent[1:2]],
                    "Tu peux maintenant me poser une question plus precise sur ce sujet, et je garde le fil.",
                )

        recent_turn = self._recent_turn_for_subject(subject_n)
        if recent_turn:
            return self._format_response(
                f"On reste sur {subject}.",
                [
                    "Je garde le contexte de notre echange precedent.",
                    f"Dernier point utile: {recent_turn.assistant[:120].strip()}",
                ],
                "Tu peux me demander un detail, une precision, ou une suite sur ce sujet.",
            )

        return self._format_response(
            f"On peut parler de {subject}.",
            [
                "Je garde ce sujet pour la suite de la conversation.",
                "Pose-moi ensuite une question plus precise: sa vie, sa carriere, ses œuvres, son histoire, ou un point particulier.",
            ],
            "Je te reponds dans le meme fil tant qu'on reste sur ce theme.",
        )

    def _answer_followup_on_subject(self, message: str, subject: str) -> str:
        subject_n = normalize(subject)
        expanded_subject = self._expand_followup_subject_query(subject_n, message)
        relation_keywords = self._followup_relation_keywords(message)
        relation_evidence = self._subject_relation_evidence(subject_n, message)
        subject_example = self._best_subject_example(expanded_subject)
        if subject_example is not None:
            best_question, best_answer, score = subject_example
            best_question_n = normalize(best_question)
            if relation_keywords and not any(keyword in best_question_n for keyword in relation_keywords):
                subject_example = None
            elif score >= 0.45:
                return self._format_response(
                    f"On reste sur {subject}.",
                    [
                        "Je reprends le contexte du sujet precedent.",
                        f"Je relie ta question a: {expanded_subject}.",
                    ],
                    best_answer,
                )

        brief = self.subject_briefs.get(subject_n, "").strip()
        if relation_keywords and not relation_evidence:
            return self._format_response(
                f"On reste sur {subject}.",
                [
                    f"Je garde le sujet {subject}, mais je n'ai pas encore d'information fiable sur {', '.join(sorted(relation_keywords))}.",
                    "Si tu me donnes un indice, un document, ou une phrase plus precise, je peux chercher avec toi.",
                ],
                "Tu peux me demander un autre detail, ou me donner le contexte exact pour ce point.",
            )

        if brief:
            if relation_evidence:
                brief = f"{brief}\nIndice deja garde: {relation_evidence}"
            return self._format_response(
                f"On reste sur {subject}.",
                [
                    brief,
                    f"Si tu parles de '{message.strip()}', je garde le lien avec {expanded_subject}.",
                ],
                "Tu peux me demander le mari, les enfants, la carriere, la date ou un autre detail.",
            )

        notes = self.subject_memory.get(subject_n, [])
        if notes:
            recent = [
                self._compact_memory_entry(note)
                for note in notes[-3:]
                if note.strip()
            ]
            if recent:
                return self._format_response(
                    f"On reste sur {subject}.",
                    [f"Ce que je retiens deja: {recent[0]}"]
                    + [f"Autre souvenir utile: {item}" for item in recent[1:2]],
                    "Tu peux me demander le point suivant, et je garde le contexte.",
                )

        recent_turn = self._recent_turn_for_subject(subject_n)
        if recent_turn:
            return self._format_response(
                f"Tu fais bien suite a {subject}.",
                [
                    "Je garde le contexte de l'echanger precedent.",
                    f"On avait deja parle de: {recent_turn.user[:90].strip()}",
                ],
                "Pose-moi la suite exacte que tu veux savoir.",
            )

        return self._format_response(
            f"Tu parles probablement de {subject}.",
            [
                "Je garde ce sujet pour la suite de la conversation.",
                f"Si tu veux, donne-moi une precision sur {expanded_subject} ou un detail precis.",
            ],
            "Je peux suivre ce fil avec toi, meme si on change de question.",
        )

    def _answer_with_subject_context(
        self,
        message: str,
        subject: str,
        related_turn: tuple[ConversationTurn, float] | None = None,
    ) -> str | None:
        subject_n = normalize(subject)
        if not subject_n:
            return None

        expanded_subject = self._expand_followup_subject_query(subject_n, message)
        relation_keywords = self._followup_relation_keywords(message)
        relation_evidence = self._subject_relation_evidence(subject_n, message)
        subject_example = self._best_subject_example(expanded_subject)
        if subject_example is not None:
            best_question, best_answer, score = subject_example
            best_question_n = normalize(best_question)
            if relation_keywords and not any(keyword in best_question_n for keyword in relation_keywords):
                subject_example = None
            elif score >= 0.3:
                return self._format_response(
                    f"Je pars du sujet {subject}.",
                    [
                        f"J'ai trouve un exemple proche: {best_question}.",
                        best_answer,
                    ],
                    "Si tu veux, je peux maintenant entrer dans un detail precis de ce sujet.",
                )

        brief = self.subject_briefs.get(subject_n, "").strip()
        if relation_keywords and not relation_evidence:
            return self._format_response(
                f"Je garde le sujet {subject}.",
                [
                    f"Je n'ai pas encore une information fiable sur {', '.join(sorted(relation_keywords))}.",
                    "Donne-moi un indice plus precis ou un document, et je le rattache a ce sujet.",
                ],
                "Je peux aussi rester sur une question plus generale du sujet si tu veux.",
            )
        if brief:
            return self._format_response(
                f"Je me base sur {subject}.",
                [
                    brief,
                    f"Je garde aussi en tete le lien avec {expanded_subject}.",
                ],
                "Pose-moi maintenant la question exacte sur ce sujet et je garde le fil.",
            )

        notes = self.subject_memory.get(subject_n, [])
        if notes:
            recent = [
                self._compact_memory_entry(note)
                for note in notes[-3:]
                if note.strip()
            ]
            if recent:
                return self._format_response(
                    f"Je me base sur {subject}.",
                    [
                        f"Ce que je retiens deja: {recent[0]}",
                        *[f"Autre point utile: {item}" for item in recent[1:2]],
                    ],
                    "Pose-moi maintenant la question exacte sur ce sujet et je garde le fil.",
                )

        if related_turn is not None:
            turn, score = related_turn
            if score >= 0.2:
                return self._format_response(
                    f"Je garde le contexte de {subject}.",
                    [
                        f"Dernier echange proche: {turn.user[:90].strip()}",
                        f"Je retenais aussi: {turn.assistant[:180].strip()}",
                    ],
                    "Tu peux enchaîner avec la question suivante, je reste sur ce sujet.",
                )

        return None

    def _next_question_hint(self, subject: str, message: str) -> str:
        subject_n = normalize(subject)
        message_n = normalize(message)
        if not subject_n:
            return "Tu peux me poser la question suivante, ou me demander de préciser un point."

        if any(word in message_n for word in ("qui est", "c'est qui", "c est qui")):
            return (
                f"Tu peux me demander par exemple: 'Quelle est la carrière de {subject} ?' "
                f"ou 'Quel lien a {subject} avec ... ?'"
            )
        if any(word in message_n for word in ("et son", "et sa", "et ses")):
            return (
                f"Tu peux demander la suite directe sur {subject}: famille, travail, date, lieu ou événement."
            )
        if any(word in message_n for word in ("comment", "pourquoi", "explique")):
            return (
                f"Tu peux me demander un détail précis sur {subject}, ou me donner une phrase plus longue à analyser."
            )
        return (
            f"Tu peux me demander la suite sur {subject}, ou me donner un sous-point précis comme la carrière, la famille ou une date."
        )

    def _response_style_hint(self, message: str) -> str:
        message_n = normalize(message)
        if any(word in message_n for word in ("court", "bref", "r?sume", "resumer", "r?sumer")):
            return "court"
        if any(word in message_n for word in ("simple", "d?butant", "debutant", "facile")):
            return "simple"
        if any(word in message_n for word in ("d?tail", "detail", "d?taill?", "detaille", "plus")):
            return "detaille"
        if any(word in message_n for word in ("?tape", "etape", "pas ? pas", "pas a pas", "checklist")):
            return "etapes"
        return "normal"

    def _question_response_frame(self, kind: str, subject: str, message: str) -> tuple[str, str | None, str | None]:
        subject_n = normalize(subject)
        subject_label = subject.strip() or "ce sujet"
        follow_up = self._next_question_hint(subject, message) if subject_n or message else None
        style = self._response_style_hint(message)
        if style == "court":
            prefix = "Version courte : "
        elif style == "simple":
            prefix = "Version simple : "
        elif style == "detaille":
            prefix = "Version d?taill?e : "
        elif style == "etapes":
            prefix = "Version en ?tapes : "
        else:
            prefix = "Id?e cl? : "

        if kind == "followup":
            return (
                "Contexte",
                f"{prefix}je garde le fil de la discussion et je reprends le point le plus proche.",
                follow_up,
            )
        if kind == "definition":
            headline = f"D?finition : {subject_label}" if subject_n else "D?finition"
            return (
                headline,
                f"{prefix}je r?ponds d'abord simplement, puis je peux d?tailler ou donner un exemple.",
                follow_up,
            )
        if kind == "why":
            headline = f"Pourquoi {subject_label}" if subject_n else "Pourquoi"
            return (
                headline,
                f"{prefix}je donne la raison principale, puis je peux aussi expliquer les effets ou l'int?r?t.",
                follow_up,
            )
        if kind == "how":
            headline = f"Fonctionnement : {subject_label}" if subject_n else "Fonctionnement"
            return (
                headline,
                f"{prefix}je peux ensuite d?couper la r?ponse en ?tapes simples si tu veux aller plus loin.",
                follow_up,
            )
        if kind == "example":
            headline = f"Exemple : {subject_label}" if subject_n else "Exemple"
            return (
                headline,
                f"{prefix}je peux te donner un cas concret, puis un deuxi?me si tu veux comparer.",
                follow_up,
            )
        if kind == "compare":
            headline = f"Comparaison : {subject_label}" if subject_n else "Comparaison"
            return (
                headline,
                f"{prefix}je peux aussi faire un tableau des diff?rences pour rendre ?a plus lisible.",
                follow_up,
            )
        if kind == "list":
            headline = f"Liste : {subject_label}" if subject_n else "Liste"
            return (
                headline,
                f"{prefix}je peux r?duire ?a ? l'essentiel ou d?velopper point par point.",
                follow_up,
            )
        if kind == "who":
            headline = f"Qui est {subject_label}" if subject_n else "Qui est-ce ?"
            return (
                headline,
                f"{prefix}je pars du r?le, de la biographie ou du contexte le plus utile.",
                follow_up,
            )
        if kind == "when":
            headline = f"Rep?re temporel : {subject_label}" if subject_n else "Rep?re temporel"
            return (
                headline,
                f"{prefix}je peux pr?ciser une date, une p?riode ou remettre les faits dans l'ordre.",
                follow_up,
            )
        if kind == "where":
            headline = f"Rep?re spatial : {subject_label}" if subject_n else "Rep?re spatial"
            return (
                headline,
                f"{prefix}je peux pr?ciser le lieu, le cadre ou l'endroit exact selon le contexte.",
                follow_up,
            )
        if kind == "count":
            headline = f"Nombre : {subject_label}" if subject_n else "Nombre"
            return (
                headline,
                f"{prefix}je donne le nombre si le contexte est clair, sinon je pr?cise ce qu'il faut compter.",
                follow_up,
            )
        return (
            "R?ponse",
            f"{prefix}je donne une r?ponse directe, puis je peux pr?ciser si tu veux aller plus loin.",
            follow_up,
        )

    def _package_answer(
        self,
        headline: str,
        answer: str,
        *,
        subject: str = "",
        message: str = "",
        detail: str | None = None,
        follow_up: str | None = None,
    ) -> str:
        main = " ".join(answer.strip().split())
        if len(main) > 420:
            main = main[:420].rsplit(" ", 1)[0].rstrip() + "..."

        details = [main] if main else []
        if detail:
            detail_n = " ".join(detail.strip().split())
            if detail_n:
                details.append(detail_n)

        if follow_up is None:
            follow_up = self._next_question_hint(subject, message) if subject or message else None
        return self._format_response(headline, details, follow_up)

    def _answer_by_question_kind(
        self,
        message: str,
        subject: str,
        related_turn: tuple[ConversationTurn, float] | None = None,
        prediction: IntentPrediction | None = None,
        entities: EntityPrediction | None = None,
        relations: list[RelationPrediction] | None = None,
    ) -> str | None:
        message_n = normalize(message)
        if not message_n:
            return None

        kind = self._question_kind(message_n)
        subject_n = normalize(subject)
        headline, style_detail, style_follow_up = self._question_response_frame(kind, subject, message)

        if kind == "followup":
            response = self._answer_with_subject_context(message, subject, related_turn)
            if response:
                return self._package_answer(
                    headline,
                    response,
                    subject=subject_n or subject,
                    message=message,
                    detail=style_detail,
                    follow_up=style_follow_up,
                )

        if kind in {"definition", "who", "how", "why", "example", "compare", "list", "when", "where"}:
            if subject_n:
                synthetic_prompt = f"{kind} {subject_n} {message_n}"
                synthetic_answer = self._answer_from_synthetic_catalog(synthetic_prompt)
                if synthetic_answer:
                    return self._package_answer(
                        headline,
                        synthetic_answer,
                        subject=subject_n or subject,
                        message=message,
                        detail=style_detail,
                        follow_up=style_follow_up,
                    )

            common_answer = self._answer_common_question(message)
            if common_answer:
                return self._package_answer(
                    headline,
                    common_answer,
                    subject=subject_n or subject,
                    message=message,
                    detail=(self._context_note(entities, relations).strip() or style_detail),
                    follow_up=style_follow_up,
                )

            if not subject_n:
                return None
                generic_answers = {
                    "definition": "Une définition dit ce que c'est, puis elle peut ajouter à quoi ça sert. Par exemple, un serveur est un programme qui répond à des requêtes.",
                    "why": "Une réponse en 'pourquoi' explique la cause principale, puis éventuellement l'intérêt ou l'effet. Par exemple, on explique pourquoi le ciel est bleu avec la lumière.",
                    "how": "Une réponse en 'comment' marche souvent mieux en étapes: entrée, traitement, sortie. Par exemple, un serveur reçoit une requête, la traite, puis répond.",
                    "example": "Un exemple concret prend un cas réel pour montrer l'idée ou l'usage de manière simple. Par exemple, comparer un chat et un chien pour montrer deux comportements différents.",
                    "compare": "Comparer deux choses consiste à regarder leur usage, leur fonctionnement, leurs avantages et leurs limites. Par exemple, comparer Python et JavaScript.",
                    "list": "Une liste utile commence par les points principaux, puis on peut ajouter des sous-points si besoin. Par exemple, lister trois avantages du web.",
                    "who": "Pour répondre à 'qui', on peut donner le rôle, le contexte et un fait marquant. Par exemple, dire qui est Molière et pourquoi il est connu.",
                    "when": "Pour répondre à 'quand', on cherche une date, une période ou un ordre chronologique. Par exemple, situer un événement dans le temps.",
                    "where": "Pour répondre à 'où', on précise le lieu, le cadre ou l'endroit exact. Par exemple, localiser une ville ou un pays.",
                    "count": "Pour répondre à 'combien', il faut préciser ce qu'on compte: personnes, objets, étapes ou exemples. Par exemple, compter les étapes d'un programme.",
                }
                generic_answer = generic_answers.get(kind)
                if generic_answer:
                    return self._package_answer(
                        headline,
                        generic_answer,
                        subject=subject_n or subject,
                        message=message,
                        detail=style_detail,
                        follow_up=style_follow_up,
                    )

        if kind == "count":
            if subject_n:
                return self._package_answer(
                    headline,
                    f"Tu veux un nombre sur {subject}. Je peux te répondre si tu précises ce que tu veux compter: personnes, dates, éléments, étapes ou exemples.",
                    subject=subject_n or subject,
                    message=message,
                    detail=style_detail,
                    follow_up=style_follow_up,
                )

        if kind == "general" and subject_n:
            subject_example = self._best_subject_example(subject_n)
            if subject_example is not None:
                best_question, best_answer, score = subject_example
                if score >= 0.28:
                    return self._package_answer(
                        f"Je pars du sujet {subject}.",
                        f"J'ai trouve un exemple proche: {best_question}. {best_answer}",
                        subject=subject_n or subject,
                        message=message,
                        detail="Je m'appuie sur le sujet courant pour éviter de repartir à zéro.",
                        follow_up=style_follow_up,
                    )

        if prediction and prediction.label == "question":
            return self._package_answer(
                headline,
                "Donne-moi un sujet plus précis, ou reformule en demandant une définition, un exemple, une explication ou une suite.",
                subject=subject_n or subject,
                message=message,
                detail=style_detail,
                follow_up=style_follow_up,
            )

        return None

    def _apply_french_correction_rules(self, text: str) -> str:
        corrected = text.strip()
        if not corrected:
            return "Une phrase vide ne peut pas etre corrigee."

        replacements = [
            (r"(?i)\bje suis aller\b", "je suis all\u00e9"),
            (r"(?i)\bje suis all\u00e9e\b", "je suis all\u00e9e"),
            (r"(?i)\bles chat\b", "les chats"),
            (r"(?i)\bun phrase\b", "une phrase"),
            (r"(?i)\bau college\b", "au coll\u00e8ge"),
            (r"(?i)\bcollege\b", "coll\u00e8ge"),
            (r"(?i)\bca\b", "\u00e7a"),
            (r"(?i)\bsa\b", "\u00e7a"),
        ]
        for pattern, replacement in replacements:
            corrected = re.sub(pattern, replacement, corrected)

        corrected = self._spellcheck_french_words(corrected)
        corrected = corrected.strip()
        if corrected and corrected[0].islower():
            corrected = corrected[0].upper() + corrected[1:]
        if not corrected.endswith((".", "!", "?")):
            corrected += "."
        return corrected

    def _make_more_natural(self, text: str) -> str:
        corrected = " ".join(text.strip().split())
        replacements = [
            (r"(?i)\bdeja\b", "déjà"),
            (r"(?i)\bsais tu\b", "sais-tu"),
            (r"(?i)\bpeux tu\b", "peux-tu"),
            (r"(?i)\bqu est ce que\b", "qu'est-ce que"),
        ]
        for pattern, replacement in replacements:
            corrected = re.sub(pattern, replacement, corrected)
        if corrected and corrected[0].islower():
            corrected = corrected[0].upper() + corrected[1:]
        if corrected and not corrected.endswith((".", "!", "?")):
            corrected += "."
        return corrected

    def _explain_correction_changes(self, original: str, corrected: str) -> str:
        original_n = normalize(original)
        corrected_n = normalize(corrected)
        if not original_n:
            return "J'ai reformule la phrase pour la rendre correcte."
        if original_n == corrected_n:
            return "Je n'ai rien change de significatif."
        original_words = original_n.split()
        corrected_words = corrected_n.split()
        sm = SequenceMatcher(None, original_words, corrected_words)
        changed = False
        for tag, _, _, _, _ in sm.get_opcodes():
            if tag != "equal":
                changed = True
                break

        if not changed:
            return "J'ai surtout retouche la ponctuation et la mise en forme."

        if "aller" in original_n and "allé" in corrected_n:
            return "J'ai corrigé le verbe et l'accord."
        if "college" in original_n and "collège" in corrected:
            return "J'ai corrigé l'orthographe du mot et les accents."
        if "ca" in original_n or "sa" in original_n:
            return "J'ai corrigé les mots courts et les accents."
        return "J'ai corrigé l'orthographe, les accords et la ponctuation."

    def _spellcheck_french_words(self, text: str) -> str:
        if FR_SPELLCHECKER is None:
            return text

        def replace_word(match: re.Match[str]) -> str:
            original = match.group(0)
            lower = original.lower()
            if lower in FR_WORD_KEEP or len(lower) <= 2:
                return original
            if "'" in lower:
                return original
            candidate = FR_SPELLCHECKER.correction(lower) or lower
            if candidate == lower:
                return original
            if original.isupper():
                return candidate.upper()
            if original[0].isupper():
                return candidate.capitalize()
            return candidate

        return re.sub(r"\b\w+\b", replace_word, text, flags=re.UNICODE)

    def predict_knowledge(self, message: str) -> list[KnowledgeSnippet]:
        if not self.dify_client.is_ready():
            return []
        try:
            return self.dify_client.retrieve(message)
        except Exception:
            return []

    def _remember(self, user_message: str, assistant_message: str) -> None:
        self.history.append(
            ConversationTurn(
                user=user_message.strip(),
                assistant=assistant_message.strip(),
            )
        )
        if len(self.history) > MAX_HISTORY_TURNS:
            self.history = self.history[-MAX_HISTORY_TURNS:]
        self.conversation_summary = self._build_conversation_summary()
        try:
            self.save()
        except OSError:
            pass

    def _compose_local_answer(
        self,
        message: str,
        prediction: IntentPrediction | None,
        entities: EntityPrediction | None,
        relations: list[RelationPrediction] | None,
        knowledge: list[KnowledgeSnippet] | None = None,
        documents: list[KnowledgeSnippet] | None = None,
    ) -> str:
        message_n = normalize(message)
        question_kind = self._question_kind(message_n)
        contextual_snippets = self._best_context_snippets(knowledge, documents)
        has_context = bool(contextual_snippets or self._find_exact_or_partial(message_n))
        if self._is_vague_question(message) and not has_context and question_kind in {"unknown", "general"}:
            return self._clarifying_question(message_n)

        related_turn = self._best_related_turn(message)
        active_subject = self.last_subject if self._should_use_last_subject(message) else ""
        profile_answer = self._answer_by_question_kind(
            message,
            active_subject,
            related_turn,
            prediction,
            entities,
            relations,
        )
        if profile_answer:
            return profile_answer

        if related_turn is not None and self._wants_context_answer(message_n):
            turn, score = related_turn
            if score >= 0.25:
                return self._format_response(
                    "Je reprends le fil de la discussion.",
                    [
                        f"Dernier point proche: {turn.user[:90].strip()}",
                        f"Je retenais aussi: {turn.assistant[:180].strip()}",
                    ],
                    "Donne-moi la suite exacte ou je peux clarifier ce point.",
                )

        nearest = self._nearest_example(message)
        if nearest:
            best_question, best_answer, score = nearest
            if score >= 0.55:
                if score >= 0.88:
                    return best_answer
                return f"{best_answer} (proche de: {best_question})"

        if contextual_snippets and self._wants_context_answer(message_n):
            contextual_answer = self._synthesize_context_answer(
                message_n,
                prediction,
                entities,
                relations,
                knowledge,
                documents,
            )
            if contextual_answer:
                return contextual_answer

        entity_note = ""
        if entities and entities.entities:
            summary = ", ".join(f"{e.text}={e.label}" for e in entities.entities[:3])
            entity_note = f" J'ai aussi repere: {summary}."
        relation_note = ""
        if relations:
            summary = ", ".join(
                f"{rel.head}->{rel.tail}={rel.label}" for rel in relations[:3]
            )
            relation_note = f" Relations: {summary}."
        lowered = normalize(message)
        if lowered.startswith(("bonjour", "salut", "coucou")):
            return "Bonjour ! Comment puis-je t'aider ?"
        if "merci" in lowered:
            return "Avec plaisir. Je peux t'aider a apprendre, corriger, expliquer ou resumer."
        if "relation" in lowered or "lien" in lowered:
            if entities and entities.entities:
                summary = ", ".join(f"{e.text}={e.label}" for e in entities.entities[:3])
                return (
                    "Je vois les elements importants : "
                    f"{summary}. Si tu veux, on peut apprendre le lien entre eux."
                )
            return "Je peux apprendre les relations entre les elements d'une phrase. Envoie un exemple simple."
        if any(word in lowered for word in ("expliquer", "explique", "resume", "corrige", "reformule")):
            if "corrige" in lowered or "reformule" in lowered:
                self.set_pending_action("correction", message)
                return "D'accord. Envoie-moi la phrase a corriger."
            return (
                "Envoie-moi le texte ou la phrase complete, et je te ferai une reponse utile "
                "a partir du contexte."
            )
        if lowered.startswith(("qui es-tu", "c'est quoi", "qu'est-ce que", "comment")):
            return (
                "Je peux t'aider a comprendre, reformuler ou retrouver l'information. "
                "Donne-moi le sujet exact, ou colle le texte, et je te reponds plus precisement."
            )

        if not any(
            token in lowered for token in ("python", "web", "ia", "histoire", "science", "serveur", "document")
        ) and not (entities and entities.entities):
            return self._clarifying_question(message_n)

        if self._needs_follow_up(message_n):
            return self._clarifying_question(message_n)

        if prediction:
            return self._compose_prediction_answer(prediction, entities, relations)

        return (
            "Je peux deja travailler avec ce que tu m'as donne, mais j'ai besoin d'un peu "
            "plus de contexte. Envoie un texte, une question plus precise, ou utilise "
            "/teach question | reponse pour m'apprendre une meilleure reponse."
        )

    def _compose_prediction_answer(
        self,
        prediction: IntentPrediction,
        entities: EntityPrediction | None,
        relations: list[RelationPrediction] | None,
    ) -> str:
        label = prediction.label
        confidence = prediction.confidence
        context = self._context_note(entities, relations)

        if label == "question":
            lead = "Je pense que c'est une question."
        elif label == "correction":
            lead = "Je pense que tu veux une correction. Envoie la phrase complete et je la corrige."
        elif label == "demande":
            lead = "Je pense que tu fais une demande. Dis-moi exactement ce que tu veux obtenir."
        elif label == "salutation":
            lead = "Je pense que tu me salues. Bonjour !"
        elif label == "resume":
            lead = "Je pense que tu veux un resume. Envoie le texte a raccourcir."
        else:
            lead = f"Je pense que ce texte ressemble a une '{label}'."

        details: list[str] = []
        if entities and entities.entities:
            summary = ", ".join(f"{e.text}={e.label}" for e in entities.entities[:3])
            details.append(f"Elements repères : {summary}.")

        if relations:
            summary = ", ".join(
                f"{rel.head}->{rel.tail}={rel.label}" for rel in relations[:3]
            )
            details.append(f"Relations utiles : {summary}.")

        if confidence < 0.35:
            details.append("Je peux affiner si tu me donnes un peu plus de contexte.")

        return self._format_response(lead, details, context.strip())

    def _best_context_snippets(
        self,
        knowledge: list[KnowledgeSnippet] | None,
        documents: list[KnowledgeSnippet] | None,
    ) -> list[KnowledgeSnippet]:
        merged: list[KnowledgeSnippet] = []
        for item in (knowledge or [])[:3]:
            merged.append(item)
        for item in (documents or [])[:3]:
            merged.append(item)
        merged.sort(key=lambda item: item.score, reverse=True)

        deduped: list[KnowledgeSnippet] = []
        seen: set[tuple[str, str]] = set()
        for item in merged:
            key = (item.source.strip().lower(), item.content.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:4]

    def _wants_context_answer(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "?",
                "qu'est-ce que",
                "qu est-ce que",
                "que sais-tu",
                "que sais tu",
                "comment",
                "pourquoi",
                "explique",
                "resume",
                "résume",
                "corrige",
                "reformule",
                "document",
                "texte",
                "passage",
                "paragraphe",
                "cours",
                "pdf",
                "fichier",
                "note",
                "décris",
                "decris",
            )
        )

    def _synthesize_context_answer(
        self,
        message_n: str,
        prediction: IntentPrediction | None,
        entities: EntityPrediction | None,
        relations: list[RelationPrediction] | None,
        knowledge: list[KnowledgeSnippet] | None,
        documents: list[KnowledgeSnippet] | None,
    ) -> str | None:
        snippets = self._best_context_snippets(knowledge, documents)
        if not snippets:
            return None

        is_summary = self._is_summary_request(message_n)
        is_correction = self._is_correction_request(message_n)
        is_explanation = any(
            word in message_n
            for word in ("explique", "expliquer", "pourquoi", "comment", "decris", "décris")
        )

        note = self._context_note(entities, relations)
        if is_summary:
            joined = " ".join(snippet.content for snippet in snippets[:3])
            summary = self._extractive_summary(joined)
            if not summary:
                return None
            source = snippets[0].source or "le contexte"
            return f"Resume a partir de {source} : {summary}{note}"

        if is_correction:
            if len(snippets) == 1:
                snippet = snippets[0]
                excerpt = snippet.content.strip()
                if len(excerpt) > 280:
                    excerpt = excerpt[:280].rsplit(" ", 1)[0].rstrip() + "..."
                source = snippet.source or "le contexte"
                return f"Version utile d'apres {source} : {excerpt}{note}"
            joined = " ".join(snippet.content for snippet in snippets[:2])
            summary = self._extractive_summary(joined)
            return f"Je corrige et je rassemble l'essentiel : {summary}{note}"

        if is_explanation:
            first = snippets[0]
            excerpt = first.content.strip()
            if len(excerpt) > 320:
                excerpt = excerpt[:320].rsplit(" ", 1)[0].rstrip() + "..."
            source = first.source or "le contexte"
            if len(snippets) > 1:
                second = snippets[1]
                extra = second.content.strip()
                if len(extra) > 180:
                    extra = extra[:180].rsplit(" ", 1)[0].rstrip() + "..."
                return f"Voici ce que je retiens de {source} : {excerpt}\nEt aussi : {extra}{note}"
            return f"Voici ce que je retiens de {source} : {excerpt}{note}"

        first = snippets[0]
        excerpt = first.content.strip()
        if len(excerpt) > 320:
            excerpt = excerpt[:320].rsplit(" ", 1)[0].rstrip() + "..."
        source = first.source or "le contexte"
        if len(snippets) > 1:
            extra = snippets[1].content.strip()
            if len(extra) > 180:
                extra = extra[:180].rsplit(" ", 1)[0].rstrip() + "..."
            return f"D'apres {source} : {excerpt}\nComplement : {extra}{note}"
        return f"D'apres {source} : {excerpt}{note}"
    def _needs_follow_up(self, message_n: str) -> bool:
        return any(
            phrase in message_n
            for phrase in (
                "je ne sais pas",
                "je comprends pas",
                "je ne comprends pas",
                "aide",
                "question",
                "peux-tu",
                "peux tu",
                "comment faire",
                "explique",
                "corrige",
                "resume",
                "rÃ©sume",
                "qu'est-ce que",
                "qu est-ce que",
                "pourquoi",
                "comment",
                "qui est",
                "quoi",
            )
        )

    def _is_vague_question(self, message: str) -> bool:
        message_n = normalize(message)
        tokens = tokenize(message_n)
        if not message_n:
            return False
        if len(tokens) <= 2 and any(
            keyword in message_n
            for keyword in (
                "comment",
                "pourquoi",
                "quoi",
                "qui",
                "peux tu",
                "peux-tu",
                "aide",
            )
        ):
            return True
        if any(
            message_n.startswith(prefix)
            for prefix in (
                "comment faire",
                "peux-tu",
                "peux tu",
                "aide",
            )
        ):
            return True
        return False

    def _clarifying_question(self, message_n: str) -> str:
        if any(word in message_n for word in ("corrige", "reformule")):
            return self._format_response(
                "Je peux corriger ça.",
                ["Envoie-moi la phrase à corriger."],
                "Je te renverrai la version corrigée et une explication rapide.",
            )
        if any(word in message_n for word in ("resume", "rÃ©sume")):
            return self._format_response(
                "Je peux résumer ça.",
                ["Envoie-moi le texte à résumer."],
                "Je te ferai une version courte et claire.",
            )
        if any(word in message_n for word in ("comment", "comment faire", "peux-tu", "peux tu")):
            return self._format_response(
                "Je peux t'aider.",
                ["Tu veux une explication simple, un exemple, ou un pas-à-pas ?"],
                "Si tu veux, je peux aussi répondre plus court.",
            )
        if any(word in message_n for word in ("pourquoi", "quoi", "qu'est-ce que", "qu est-ce que", "qui est")):
            return self._format_response(
                "Je peux t'aider.",
                ["Dis-moi le sujet exact ou colle le texte."],
                "Je répondrai plus précisément avec le bon contexte.",
            )
        return self._format_response(
            "Je peux t'aider, mais j'ai besoin d'un peu plus de contexte.",
            ["Tu veux une explication, une correction, un résumé ou un exemple ?"],
            "Tu peux aussi me donner un mot-clé et je m'adapte.",
        )

    def _context_note(
        self,
        entities: EntityPrediction | None,
        relations: list[RelationPrediction] | None,
    ) -> str:
        entity_note = ""
        if entities and entities.entities:
            summary = ", ".join(f"{e.text}={e.label}" for e in entities.entities[:3])
            entity_note = f" J'ai aussi repere: {summary}."
        relation_note = ""
        if relations:
            summary = ", ".join(
                f"{rel.head}->{rel.tail}={rel.label}" for rel in relations[:3]
            )
            relation_note = f" Relations: {summary}."
        return f"{entity_note}{relation_note}"

    def _compose_knowledge_answer(
        self,
        message: str,
        entities: EntityPrediction | None,
        relations: list[RelationPrediction] | None,
        knowledge: list[KnowledgeSnippet],
    ) -> str:
        top = knowledge[0]
        chunk_text = top.content.strip()
        if len(chunk_text) > 320:
            chunk_text = chunk_text[:320].rsplit(" ", 1)[0].rstrip() + "..."
        prefix = "D'apres Dify"
        if top.source:
            prefix += f" ({top.source})"
        details = [chunk_text]
        context_note = self._context_note(entities, relations).strip()
        if context_note:
            details.append(context_note)
        return self._format_response(prefix, details, "Je peux chercher un autre passage si tu veux.")

    def _find_exact_or_partial(self, message_n: str) -> str | None:
        exact = self._find_exact(message_n)
        if exact:
            return exact

        best_answer = None
        best_question = None
        best_score = 0.0
        for item in self._candidate_examples(message_n):
            score = _text_similarity(message_n, item["question"])
            if score > best_score:
                best_score = score
                best_answer = item["answer"]
                best_question = item["question"]
        overlap = _token_overlap_score(message_n, best_question or "")
        if best_score >= 0.9 or (best_score >= 0.72 and overlap >= 2):
            if best_question and best_answer:
                return best_answer
        return None

    def _nearest_example(self, message: str) -> tuple[str, str, float] | None:
        message_n = normalize(message)
        best: tuple[str, str, float] | None = None
        for item in self._candidate_examples(message_n):
            score = _text_similarity(message_n, item["question"])
            candidate = (item["question"], item["answer"], score)
            if best is None or score > best[2]:
                best = candidate
        return best

    def _find_exact(self, message_n: str) -> str | None:
        return self.example_index.get(message_n)

    def _candidate_examples(self, message_n: str, limit: int = 800) -> list[dict[str, str]]:
        tokens = [
            token
            for token in tokenize(message_n)
            if len(token) > 2 and token not in EXAMPLE_INDEX_STOP_WORDS
        ]
        if not tokens:
            return self.examples[:limit]

        hits: Counter[int] = Counter()
        for token in tokens:
            for index in self.example_token_index.get(token, set()):
                hits[index] += 1

        if not hits:
            return []

        return [
            self.examples[index]
            for index, _ in hits.most_common(limit)
            if 0 <= index < len(self.examples)
        ]

    def _clean_answer(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^\s*Assistant:\s*", "", cleaned, flags=re.IGNORECASE)
        return self._polish_output_text(cleaned.strip())

    def _polish_output_text(self, text: str) -> str:
        replacements = {
            "RÃ©ponse": "Reponse",
            "Réponse": "Reponse",
            "rÃ©ponse": "reponse",
            "réponse": "reponse",
            "R?ponse": "Reponse",
            "r?ponse": "reponse",
            "D?finition": "Definition",
            "d?finition": "definition",
            "Id?e cl?": "Idee cle",
            "id?e cl?": "idee cle",
            "pr?cis": "precis",
            "pr?cise": "precise",
            "pr?ciser": "preciser",
            "d?tail": "detail",
            "d?tails": "details",
            "d?tailler": "detailler",
            "?tape": "etape",
            "?tapes": "etapes",
            "r?ponds": "reponds",
            "r?pondre": "repondre",
            "s?r": "sur",
            "Ã©": "e",
            "Ã¨": "e",
            "Ãª": "e",
            "Ã ": "a",
            "Ã§": "c",
            "Â·": "-",
        }
        polished = text
        for bad, good in replacements.items():
            polished = polished.replace(bad, good)
        return polished

    def _format_response(
        self,
        headline: str,
        details: list[str] | None = None,
        follow_up: str | None = None,
    ) -> str:
        parts = [headline.strip()]
        for item in details or []:
            item_n = " ".join(str(item).strip().split())
            if item_n:
                parts.append(item_n)
        if follow_up:
            follow_n = " ".join(str(follow_up).strip().split())
            if follow_n:
                parts.append(follow_n)
        return self._polish_output_text("\n".join(parts).strip())

    def _frame_answer(
        self,
        answer: str,
        *,
        headline: str = "Réponse",
        detail: str | None = None,
        follow_up: str | None = None,
    ) -> str:
        cleaned = " ".join(answer.strip().split())
        if not cleaned:
            return headline.strip()
        details = [cleaned]
        if detail:
            details.append(detail)
        return self._format_response(headline, details, follow_up)


def _load_examples(raw_examples: Any) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    if not isinstance(raw_examples, list):
        return [item.copy() for item in DEFAULT_EXAMPLES]
    for item in raw_examples:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if not question or not answer:
            continue
        examples.append(
            {"question": normalize(str(question)), "answer": str(answer).strip()}
        )
    return examples or [item.copy() for item in DEFAULT_EXAMPLES]


def _load_history(raw_history: Any) -> list[ConversationTurn]:
    history: list[ConversationTurn] = []
    if not isinstance(raw_history, list):
        return history
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        user = str(item.get("user", "")).strip()
        assistant = str(item.get("assistant", "")).strip()
        if not user or not assistant:
            continue
        history.append(ConversationTurn(user=user, assistant=assistant))
    return history[-MAX_HISTORY_TURNS:]


def _load_memory_notes(raw_notes: Any) -> list[str]:
    notes: list[str] = []
    if not isinstance(raw_notes, list):
        return notes
    for item in raw_notes:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.strip().split())
        if cleaned:
            notes.append(cleaned)
    return notes[-20:]


def _load_memory_sources(raw_memory_sources: Any) -> dict[str, list[str]]:
    if not isinstance(raw_memory_sources, dict):
        return {}
    memory_sources: dict[str, list[str]] = {}
    for key, value in raw_memory_sources.items():
        source = normalize(str(key))
        if not source:
            continue
        notes: list[str] = []
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, str):
                    continue
                cleaned = _strip_memory_markers(item)
                if cleaned and cleaned not in notes:
                    notes.append(cleaned)
        if notes:
            memory_sources[source] = notes[-20:]
    return memory_sources


def _load_documents(raw_documents: Any) -> list[DocumentMemory]:
    documents: list[DocumentMemory] = []
    if not isinstance(raw_documents, list):
        return documents
    for item in raw_documents:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip() or f"Document {len(documents) + 1}"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        documents.append(DocumentMemory(title=title, content=content))
    return documents[-20:]


def _load_preferences(raw_preferences: Any) -> dict[str, str]:
    if not isinstance(raw_preferences, dict):
        return {}
    preferences: dict[str, str] = {}
    for key, value in raw_preferences.items():
        key_n = normalize(str(key))
        value_n = " ".join(str(value).strip().split())
        if key_n and value_n:
            preferences[key_n] = value_n
    return preferences


def _load_subject_memory(raw_subject_memory: Any) -> dict[str, list[str]]:
    if not isinstance(raw_subject_memory, dict):
        return {}
    subject_memory: dict[str, list[str]] = {}
    for key, value in raw_subject_memory.items():
        subject = normalize(str(key))
        if not subject:
            continue
        notes: list[str] = []
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, str):
                    continue
                cleaned = " ".join(item.strip().split())
                if cleaned and cleaned not in notes:
                    notes.append(cleaned)
        if notes:
            subject_memory[subject] = notes[-10:]
    return subject_memory


def _load_subject_briefs(raw_subject_briefs: Any) -> dict[str, str]:
    if not isinstance(raw_subject_briefs, dict):
        return {}
    subject_briefs: dict[str, str] = {}
    for key, value in raw_subject_briefs.items():
        subject = normalize(str(key))
        brief = _strip_memory_markers(str(value))
        if subject and brief:
            if len(brief) > 500:
                brief = brief[:500].rsplit(" ", 1)[0].rstrip() + "..."
            subject_briefs[subject] = brief
    return subject_briefs


def _strip_memory_markers(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ""
    for marker in (
        "Sujet:",
        "Points récents:",
        "Points recents:",
        "Dernier echange:",
        "Dernier échange:",
        "Fils de sujet:",
        "Sources:",
        "Preferences:",
        "Préférences:",
        "Sujets frequents:",
        "Sujets fréquents:",
        "Sujet courant:",
    ):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip(" .|")
    return cleaned


def _load_training_samples(base_dir: Path) -> list[dict[str, str]]:
    training_file = base_dir / TRAINING_FILE_NAME
    if not training_file.exists():
        return []
    try:
        raw = json.loads(training_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    samples: list[dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            label = item.get("label")
            if not text or not label:
                continue
            samples.append({"text": str(text), "label": str(label)})
    return samples


def _load_entity_samples(base_dir: Path) -> list[dict[str, Any]]:
    ner_file = base_dir / NER_FILE_NAME
    if not ner_file.exists():
        return []
    try:
        raw = json.loads(ner_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return raw if isinstance(raw, list) else []


def _load_qa_seed(base_dir: Path) -> list[dict[str, str]]:
    seed_file = base_dir / QA_SEED_FILE_NAME
    if not seed_file.exists():
        return []
    try:
        raw = json.loads(seed_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return _load_examples(raw)


def _load_qa_model(base_dir: Path) -> list[dict[str, str]]:
    model_file = base_dir / QA_MODEL_FILE_NAME
    if not model_file.exists():
        return []
    try:
        raw = json.loads(model_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    return _load_examples(raw.get("examples", []))


def _load_science_examples(base_dir: Path) -> list[dict[str, str]]:
    science_file = base_dir / SCIENCE_FILE_NAME
    if not science_file.exists():
        return []
    try:
        raw = json.loads(science_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return _load_examples(raw)


def _load_extra_qa_examples(base_dir: Path) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    for file_name in EXTRA_QA_FILE_NAMES:
        path = base_dir / file_name
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        collected.extend(_load_examples(raw))
    return collected


def _merge_examples(*groups: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[str, str] = {}
    for group in groups:
        for item in group:
            question = normalize(item.get("question", ""))
            answer = item.get("answer", "").strip()
            if question and answer and question not in merged:
                merged[question] = answer
    return [{"question": q, "answer": a} for q, a in merged.items()]


def _load_intent_classifier(base_dir: Path) -> IntentClassifier:
    samples = _load_training_samples(base_dir)
    if not samples:
        return IntentClassifier()
    return IntentClassifier.train(samples)


def _load_entity_extractor(base_dir: Path) -> EntityExtractor:
    samples = _load_entity_samples(base_dir)
    if not samples:
        return EntityExtractor()
    return EntityExtractor.train(samples)


def _load_relation_samples(base_dir: Path) -> list[dict[str, Any]]:
    relation_file = base_dir / RELATIONS_FILE_NAME
    if not relation_file.exists():
        return []
    try:
        raw = json.loads(relation_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return raw if isinstance(raw, list) else []


def _load_relation_extractor(base_dir: Path) -> RelationExtractor:
    samples = _load_relation_samples(base_dir)
    if not samples:
        return RelationExtractor()
    return RelationExtractor.train(samples)


def _load_dify_client() -> DifyKnowledgeClient:
    return DifyKnowledgeClient()


def _token_overlap_score(a: str, b: str) -> int:
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    return len(tokens_a & tokens_b)


def _text_similarity(a: str, b: str) -> float:
    a_n = normalize(a)
    b_n = normalize(b)
    if not a_n or not b_n:
        return 0.0
    ratio = SequenceMatcher(None, a_n, b_n).ratio()
    tokens_a = set(tokenize(a_n))
    tokens_b = set(tokenize(b_n))
    if tokens_a and tokens_b:
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    else:
        jaccard = 0.0
    return (ratio * 0.7) + (jaccard * 0.3)


def _softmax_confidence(scores: dict[str, float], best_label: str) -> float:
    best = scores[best_label]
    denom = sum(math.exp(score - best) for score in scores.values())
    if denom <= 0:
        return 0.0
    return 1.0 / denom


def _cleanup_surface(text: str) -> str:
    return text.strip().strip(".,;:!? ")


def _find_phrase_occurrences(text: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase:
        return []
    pattern = re.escape(phrase)
    matches = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        matches.append((match.start(), match.end()))
    if matches:
        return matches

    stripped = phrase.rstrip(".,;:!? ")
    if stripped != phrase:
        pattern = re.escape(stripped)
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((match.start(), match.end()))
    return matches


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


def _relation_signature(text: str, head: str, tail: str) -> str | None:
    head_hit = _find_phrase_occurrences(text, head)
    tail_hit = _find_phrase_occurrences(text, tail)
    if not head_hit or not tail_hit:
        return None
    return _signature_between(text, head_hit[0], tail_hit[0])


def _relation_signature_from_entities(
    text: str,
    head: EntitySpan,
    tail: EntitySpan,
) -> str | None:
    return _signature_between(text, (head.start, head.end), (tail.start, tail.end))


def _signature_between(text: str, left: tuple[int, int], right: tuple[int, int]) -> str | None:
    if left[1] <= right[0]:
        between = text[left[1] : right[0]]
        direction = "forward"
    elif right[1] <= left[0]:
        between = text[right[1] : left[0]]
        direction = "backward"
    else:
        return None
    tokens = tokenize(between)
    signature = " ".join(tokens[:4])
    return f"{direction}:{signature}" if signature else direction

