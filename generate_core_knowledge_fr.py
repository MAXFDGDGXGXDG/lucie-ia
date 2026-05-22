from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("lucie_core_knowledge_fr.json")


TOPICS: dict[str, dict[str, str]] = {
    "intelligence artificielle": {
        "definition": "L'intelligence artificielle regroupe des programmes capables d'analyser des donnees, reconnaitre des motifs, generer du texte, classer des informations ou aider a prendre des decisions.",
        "utilite": "Elle sert a automatiser des taches, assister la recherche, expliquer des documents, programmer, traduire, resumer et dialoguer avec un utilisateur.",
        "fonctionnement": "Une IA apprend souvent a partir d'exemples: elle repere des regularites, puis les reutilise pour produire une reponse probable.",
        "limite": "Une IA peut se tromper, inventer une reponse ou manquer de contexte. Il faut verifier les informations importantes.",
        "exemple": "Un assistant qui resume un texte, un modele qui reconnait une image ou un programme qui classe des emails sont des exemples d'IA.",
    },
    "modele de langage": {
        "definition": "Un modele de langage predit et produit du texte a partir d'un contexte.",
        "utilite": "Il peut expliquer, reformuler, coder, corriger, resumer et tenir une conversation.",
        "fonctionnement": "Il transforme les mots en representations numeriques, tient compte du contexte, puis genere la suite la plus utile.",
        "limite": "Il n'a pas toujours acces au monde reel et peut confondre des faits proches.",
        "exemple": "ChatGPT et Lucie sont des assistants bases sur des principes de modele de langage.",
    },
    "apprentissage automatique": {
        "definition": "L'apprentissage automatique consiste a entrainer un programme avec des donnees pour qu'il generalise sur de nouveaux cas.",
        "utilite": "Il sert a predire, classer, detecter des anomalies et recommander des actions.",
        "fonctionnement": "On donne des exemples, on mesure l'erreur, puis on ajuste le modele pour reduire cette erreur.",
        "limite": "Un modele apprend les biais et les limites de ses donnees d'entrainement.",
        "exemple": "Predire le prix d'une maison avec sa surface et sa localisation est un exemple classique.",
    },
    "python": {
        "definition": "Python est un langage de programmation lisible et polyvalent.",
        "utilite": "Il sert a creer des scripts, sites web, IA, robots, analyses de donnees et outils d'automatisation.",
        "fonctionnement": "On ecrit des instructions dans un fichier, l'interpreteur Python les execute ligne par ligne.",
        "limite": "Python est parfois moins rapide que C ou Rust pour du calcul tres bas niveau.",
        "exemple": "print('Bonjour') affiche Bonjour dans le terminal.",
    },
    "fonction python": {
        "definition": "Une fonction Python est un bloc de code reutilisable qui peut recevoir des parametres et renvoyer un resultat.",
        "utilite": "Elle evite de recopier du code et rend un programme plus clair.",
        "fonctionnement": "On la declare avec def, puis on l'appelle avec son nom et des parentheses.",
        "limite": "Une fonction doit avoir un role clair; sinon le code devient difficile a maintenir.",
        "exemple": "def addition(a, b): return a + b",
    },
    "api": {
        "definition": "Une API est une interface qui permet a deux programmes de communiquer.",
        "utilite": "Elle sert a envoyer des donnees, demander une action ou recuperer une reponse depuis un service.",
        "fonctionnement": "Un client envoie une requete a une adresse, le serveur traite la demande et renvoie une reponse.",
        "limite": "Une API doit etre protegee, documentee et controlee pour eviter les abus.",
        "exemple": "Une application meteo utilise souvent une API pour recuperer la temperature.",
    },
    "serveur web": {
        "definition": "Un serveur web est un programme qui recoit des requetes HTTP et renvoie des pages ou des donnees.",
        "utilite": "Il permet a un site ou une application d'etre accessible depuis un navigateur.",
        "fonctionnement": "Le navigateur demande une ressource, le serveur calcule ou lit la reponse, puis l'envoie.",
        "limite": "Un serveur doit gerer la securite, les erreurs, les utilisateurs et la charge.",
        "exemple": "Render heberge Lucie en ligne avec un serveur web.",
    },
    "html": {
        "definition": "HTML decrit la structure d'une page web.",
        "utilite": "Il sert a organiser les titres, paragraphes, boutons, formulaires, images et liens.",
        "fonctionnement": "Le navigateur lit les balises HTML et construit la page.",
        "limite": "HTML seul ne gere ni le style avance ni la logique interactive.",
        "exemple": "<button>Envoyer</button> cree un bouton.",
    },
    "css": {
        "definition": "CSS sert a styliser une page web.",
        "utilite": "Il controle les couleurs, espacements, tailles, polices, grilles et animations.",
        "fonctionnement": "On cible des elements HTML avec des selecteurs puis on leur applique des proprietes.",
        "limite": "Un style trop complique devient difficile a maintenir.",
        "exemple": "button { color: white; background: black; } stylise les boutons.",
    },
    "javascript": {
        "definition": "JavaScript est un langage qui rend les pages web interactives.",
        "utilite": "Il sert a reagir aux clics, envoyer des requetes, modifier l'interface et creer des apps web.",
        "fonctionnement": "Le navigateur execute le code JavaScript et peut modifier la page en direct.",
        "limite": "Un mauvais JavaScript peut rendre une page lente ou difficile a deboguer.",
        "exemple": "document.querySelector('button') selectionne un bouton dans la page.",
    },
    "base de donnees": {
        "definition": "Une base de donnees stocke des informations organisees.",
        "utilite": "Elle permet de retrouver, modifier, filtrer et relier des donnees.",
        "fonctionnement": "On envoie des requetes pour lire ou ecrire des donnees.",
        "limite": "Il faut sauvegarder, securiser et structurer correctement les donnees.",
        "exemple": "Une table utilisateurs peut contenir nom, email et date de creation.",
    },
    "algorithme": {
        "definition": "Un algorithme est une suite d'etapes pour resoudre un probleme.",
        "utilite": "Il permet de rendre une solution precise, reproductible et programmable.",
        "fonctionnement": "On definit les entrees, les operations et la sortie attendue.",
        "limite": "Un mauvais algorithme peut etre lent ou donner un mauvais resultat.",
        "exemple": "Trier une liste du plus petit au plus grand est un algorithme.",
    },
    "complexite algorithmique": {
        "definition": "La complexite mesure le temps ou la memoire necessaire quand la taille du probleme augmente.",
        "utilite": "Elle aide a choisir une solution efficace.",
        "fonctionnement": "On decrit souvent la complexite avec une notation comme O(n), O(log n) ou O(n^2).",
        "limite": "La notation donne une tendance, pas toujours le temps exact sur une machine.",
        "exemple": "Parcourir une liste une fois est souvent O(n).",
    },
    "robotique": {
        "definition": "La robotique combine mecanique, electronique, informatique et capteurs pour creer des machines capables d'agir.",
        "utilite": "Elle sert a automatiser des mouvements, explorer, aider, fabriquer ou apprendre.",
        "fonctionnement": "Un controleur lit des capteurs, decide une action, puis envoie des ordres aux moteurs.",
        "limite": "La mecanique, l'alimentation et la securite sont souvent aussi importantes que le code.",
        "exemple": "Un robot mobile peut utiliser des moteurs, une batterie, un microcontroleur et des capteurs de distance.",
    },
    "servo moteur": {
        "definition": "Un servomoteur est un moteur controle en position.",
        "utilite": "Il sert a orienter un bras, une roue directrice, une camera ou une articulation.",
        "fonctionnement": "On lui envoie un signal PWM qui indique l'angle voulu.",
        "limite": "Il faut respecter le couple, la tension et le courant necessaires.",
        "exemple": "Un servo peut tourner une tete de robot a gauche ou a droite.",
    },
    "raspberry pi": {
        "definition": "Un Raspberry Pi est un petit ordinateur monocarte.",
        "utilite": "Il sert a faire tourner Linux, des scripts Python, des serveurs locaux, de la vision ou de la robotique.",
        "fonctionnement": "Il execute un systeme d'exploitation et communique avec des composants via USB, GPIO, I2C ou Wi-Fi.",
        "limite": "Ses broches GPIO ne doivent pas alimenter directement de gros moteurs.",
        "exemple": "Un Raspberry Pi peut piloter un PCA9685 qui controle des servos.",
    },
    "pca9685": {
        "definition": "Le PCA9685 est un controleur PWM souvent utilise pour piloter plusieurs servomoteurs.",
        "utilite": "Il permet de commander jusqu'a 16 sorties PWM avec peu de broches du Raspberry Pi.",
        "fonctionnement": "Le Raspberry Pi communique avec lui en I2C, puis le module genere les signaux PWM.",
        "limite": "Les servos doivent avoir une alimentation separee assez puissante.",
        "exemple": "Un robot avec 12 servos peut utiliser un PCA9685 pour centraliser le controle.",
    },
    "batterie lipo": {
        "definition": "Une batterie LiPo est une batterie lithium-polymere legere et capable de fournir beaucoup de courant.",
        "utilite": "Elle est souvent utilisee en robotique, drones et modelisme.",
        "fonctionnement": "Elle fournit une tension selon le nombre de cellules, par exemple 2S vaut environ 7,4 V nominal.",
        "limite": "Elle doit etre chargee avec un chargeur adapte et manipulee avec prudence.",
        "exemple": "Une LiPo 2S peut alimenter des servos via un regulateur adapte.",
    },
    "electricite": {
        "definition": "L'electricite decrit le deplacement de charges electriques.",
        "utilite": "Elle alimente les circuits, moteurs, ordinateurs et capteurs.",
        "fonctionnement": "La tension pousse le courant dans un circuit, tandis que la resistance limite ce courant.",
        "limite": "Une mauvaise alimentation peut chauffer, abimer des composants ou etre dangereuse.",
        "exemple": "La loi d'Ohm s'ecrit U = R x I.",
    },
    "loi d'ohm": {
        "definition": "La loi d'Ohm relie tension, resistance et courant: U = R x I.",
        "utilite": "Elle sert a calculer le courant dans un circuit simple.",
        "fonctionnement": "Si la tension augmente, le courant augmente; si la resistance augmente, le courant diminue.",
        "limite": "Elle s'applique surtout aux composants ohmiques simples.",
        "exemple": "Avec 5 V et 100 ohms, le courant vaut 0,05 A.",
    },
    "mathematiques": {
        "definition": "Les mathematiques etudient les nombres, formes, structures, relations et raisonnements.",
        "utilite": "Elles servent en science, informatique, economie, robotique et logique.",
        "fonctionnement": "On part de definitions et de regles pour demontrer ou calculer.",
        "limite": "Un resultat mathematique doit etre justifie par un raisonnement clair.",
        "exemple": "Resoudre une equation consiste a trouver les valeurs qui la rendent vraie.",
    },
    "fraction": {
        "definition": "Une fraction represente une partie d'un tout ou un quotient.",
        "utilite": "Elle sert a comparer, partager et calculer des proportions.",
        "fonctionnement": "Le numerateur indique combien de parts on prend, le denominateur combien de parts composent le tout.",
        "limite": "Le denominateur ne peut pas etre zero.",
        "exemple": "1/2 represente une moitie.",
    },
    "probabilite": {
        "definition": "La probabilite mesure la chance qu'un evenement se produise.",
        "utilite": "Elle sert a raisonner dans l'incertitude.",
        "fonctionnement": "Une probabilite est souvent comprise entre 0 et 1.",
        "limite": "Une probabilite ne garantit pas ce qui arrivera dans un cas unique.",
        "exemple": "Avec une piece equilibree, la probabilite de pile est 1/2.",
    },
    "statistiques": {
        "definition": "Les statistiques permettent de collecter, resumer et interpreter des donnees.",
        "utilite": "Elles servent a comprendre des tendances et comparer des groupes.",
        "fonctionnement": "On utilise des mesures comme moyenne, mediane, variance et correlation.",
        "limite": "Correlation ne veut pas dire causalite.",
        "exemple": "La moyenne d'une serie est la somme des valeurs divisee par leur nombre.",
    },
    "physique": {
        "definition": "La physique etudie la matiere, l'energie, le mouvement et les forces.",
        "utilite": "Elle explique des phenomenes comme la gravite, la lumiere, l'electricite et la chaleur.",
        "fonctionnement": "Elle utilise des observations, mesures, modeles et equations.",
        "limite": "Un modele physique est une approximation utile du reel.",
        "exemple": "La force peut modifier le mouvement d'un objet.",
    },
    "gravite": {
        "definition": "La gravite est une interaction qui attire les masses entre elles.",
        "utilite": "Elle explique la chute des objets, les orbites et le poids.",
        "fonctionnement": "Plus deux masses sont grandes et proches, plus l'attraction est forte.",
        "limite": "Pour des cas extremes, il faut utiliser la relativite generale.",
        "exemple": "La Terre attire les objets vers son centre.",
    },
    "chimie": {
        "definition": "La chimie etudie la matiere, les atomes, les molecules et leurs transformations.",
        "utilite": "Elle sert a comprendre les reactions, materiaux, medicaments et aliments.",
        "fonctionnement": "Les atomes se lient et se rearrangent pendant les reactions chimiques.",
        "limite": "Certaines reactions peuvent etre dangereuses sans precautions.",
        "exemple": "L'eau est une molecule composee de deux atomes d'hydrogene et un d'oxygene.",
    },
    "biologie": {
        "definition": "La biologie etudie les etres vivants.",
        "utilite": "Elle explique les cellules, organismes, ecosystemes, evolution et sante.",
        "fonctionnement": "Elle observe les structures vivantes et les processus comme reproduction, respiration et heredite.",
        "limite": "Le vivant est complexe et depend souvent du contexte.",
        "exemple": "La cellule est une unite de base du vivant.",
    },
    "adn": {
        "definition": "L'ADN est une molecule qui porte une partie de l'information genetique des etres vivants.",
        "utilite": "Il sert a transmettre des caracteres et guider la fabrication de molecules utiles a la cellule.",
        "fonctionnement": "Il est compose de sequences de bases chimiques organisees en genes et regions regulatrices.",
        "limite": "Un caractere depend souvent de plusieurs genes et de l'environnement.",
        "exemple": "La couleur des yeux depend de facteurs genetiques multiples.",
    },
    "histoire": {
        "definition": "L'histoire etudie les evenements passes et leurs traces.",
        "utilite": "Elle aide a comprendre les societes, les conflits, les idees et les changements.",
        "fonctionnement": "Les historiens comparent des sources et replacent les faits dans leur contexte.",
        "limite": "Une source peut etre incomplete, partiale ou difficile a interpreter.",
        "exemple": "Etudier la Revolution francaise demande de comprendre la societe de l'epoque.",
    },
    "geographie": {
        "definition": "La geographie etudie les espaces, les territoires et les relations entre humains et milieux.",
        "utilite": "Elle sert a comprendre les villes, paysages, climats, populations et ressources.",
        "fonctionnement": "Elle utilise des cartes, donnees, observations et analyses spatiales.",
        "limite": "Un territoire change avec le temps et selon les activites humaines.",
        "exemple": "Une carte peut montrer la densite de population d'un pays.",
    },
    "francais": {
        "definition": "Le francais est une langue romane utilisee pour communiquer, argumenter, raconter et expliquer.",
        "utilite": "Bien maitriser le francais aide a comprendre des textes et exprimer une idee clairement.",
        "fonctionnement": "La langue utilise grammaire, conjugaison, vocabulaire, syntaxe et ponctuation.",
        "limite": "Le sens d'une phrase depend souvent du contexte.",
        "exemple": "Une phrase claire a generalement un sujet, un verbe et un complement.",
    },
    "resume": {
        "definition": "Un resume reformule l'essentiel d'un texte en plus court.",
        "utilite": "Il aide a retenir les idees importantes sans garder tous les details.",
        "fonctionnement": "On repere le sujet, les idees principales, puis on retire exemples et repetitions.",
        "limite": "Un resume ne doit pas deformer le sens du texte.",
        "exemple": "Pour resumer un article, on garde le probleme, la methode et la conclusion.",
    },
    "argumentation": {
        "definition": "L'argumentation consiste a defendre une idee avec des raisons et des exemples.",
        "utilite": "Elle sert a convaincre, expliquer ou debattre.",
        "fonctionnement": "On formule une these, puis on donne des arguments et des preuves.",
        "limite": "Un argument faible peut etre contredit par un contre-exemple.",
        "exemple": "Dire qu'une solution est utile puis montrer un cas concret renforce l'argument.",
    },
    "methode de travail": {
        "definition": "Une methode de travail est une organisation pour apprendre ou realiser une tache efficacement.",
        "utilite": "Elle evite de se disperser et permet de progresser plus vite.",
        "fonctionnement": "On definit un objectif, on decoupe en etapes, on teste, puis on corrige.",
        "limite": "Une methode doit s'adapter a la personne et au contexte.",
        "exemple": "Pour reviser, on peut faire fiche courte, exercice, correction, puis rappel le lendemain.",
    },
    "securite informatique": {
        "definition": "La securite informatique protege les systemes, comptes et donnees.",
        "utilite": "Elle evite le vol de donnees, les intrusions et les pertes.",
        "fonctionnement": "On utilise mots de passe solides, mises a jour, sauvegardes, permissions et chiffrement.",
        "limite": "Aucune protection n'est parfaite; il faut reduire les risques.",
        "exemple": "Ne jamais publier une cle API dans un depot GitHub public.",
    },
    "confidentialite": {
        "definition": "La confidentialite consiste a controler qui peut voir une information.",
        "utilite": "Elle protege la vie privee, les identifiants et les documents sensibles.",
        "fonctionnement": "On limite l'acces, on evite les partages inutiles et on masque les secrets.",
        "limite": "Une donnee publiee peut etre difficile a retirer completement.",
        "exemple": "Un fichier contenant un mot de passe doit rester local et ignore par Git.",
    },
    "git": {
        "definition": "Git est un outil de gestion de versions.",
        "utilite": "Il permet de suivre les modifications, revenir en arriere et collaborer.",
        "fonctionnement": "On modifie des fichiers, on les stage, puis on cree un commit.",
        "limite": "Il ne faut pas committer de secrets ou fichiers inutiles.",
        "exemple": "git status montre les fichiers modifies.",
    },
    "github": {
        "definition": "GitHub est une plateforme pour heberger des depots Git.",
        "utilite": "Elle sert a partager du code, suivre des bugs et deployer des projets.",
        "fonctionnement": "On pousse une branche vers GitHub, puis on peut ouvrir une pull request ou deployer.",
        "limite": "Un depot public rend le code visible par tous.",
        "exemple": "Lucie est deployee depuis un depot GitHub vers Render.",
    },
    "render": {
        "definition": "Render est une plateforme d'hebergement d'applications web.",
        "utilite": "Elle permet de mettre une application en ligne depuis un depot Git.",
        "fonctionnement": "Render reconstruit le projet apres un push et lance le serveur.",
        "limite": "Les offres gratuites peuvent dormir, redemarrer ou etre plus lentes.",
        "exemple": "Une app Python peut etre lancee sur Render avec un Procfile.",
    },
    "raisonnement": {
        "definition": "Le raisonnement consiste a relier des idees pour arriver a une conclusion.",
        "utilite": "Il aide a comprendre, prouver, choisir et resoudre des problemes.",
        "fonctionnement": "On part d'informations fiables, on applique des regles, puis on verifie la conclusion.",
        "limite": "Une mauvaise hypothese peut mener a une mauvaise conclusion.",
        "exemple": "Si tous les servos consomment beaucoup, il faut prevoir une alimentation assez puissante.",
    },
    "resolution de probleme": {
        "definition": "Resoudre un probleme consiste a comprendre l'objectif, identifier les contraintes et tester une solution.",
        "utilite": "C'est utile en code, sciences, bricolage, robotique et organisation.",
        "fonctionnement": "On observe, formule une hypothese, agit, mesure, puis ajuste.",
        "limite": "Essayer au hasard sans mesurer rend le diagnostic difficile.",
        "exemple": "Si un robot n'avance pas, on teste alimentation, code, branchements, puis moteur.",
    },
}

QUESTION_FORMS = {
    "definition": [
        "qu'est-ce que {topic}",
        "c'est quoi {topic}",
        "definis {topic}",
        "explique {topic} simplement",
    ],
    "utilite": [
        "a quoi sert {topic}",
        "pourquoi utiliser {topic}",
        "quelle est l'utilite de {topic}",
    ],
    "fonctionnement": [
        "comment fonctionne {topic}",
        "comment marche {topic}",
        "explique le fonctionnement de {topic}",
    ],
    "limite": [
        "quelles sont les limites de {topic}",
        "a quoi faire attention avec {topic}",
        "quels sont les risques de {topic}",
    ],
    "exemple": [
        "donne un exemple de {topic}",
        "exemple simple de {topic}",
        "montre un cas concret de {topic}",
    ],
}


def build_answer(topic: str, kind: str, facts: dict[str, str]) -> str:
    detail = facts[kind]
    return (
        f"Sujet: {topic}\n"
        f"Reponse: {detail}\n"
        f"A retenir: {facts['definition']}\n"
        f"Exemple: {facts['exemple']}"
    )


def main() -> None:
    examples: list[dict[str, str]] = []
    for topic, facts in TOPICS.items():
        for kind, questions in QUESTION_FORMS.items():
            for question in questions:
                examples.append(
                    {
                        "question": question.format(topic=topic),
                        "answer": build_answer(topic, kind, facts),
                    }
                )
        examples.append(
            {
                "question": f"resume {topic}",
                "answer": (
                    f"Resume de {topic}: {facts['definition']} "
                    f"Utilite: {facts['utilite']} "
                    f"Point de vigilance: {facts['limite']}"
                ),
            }
        )

    OUTPUT.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OK: {len(examples)} connaissances ecrites dans {OUTPUT}")


if __name__ == "__main__":
    main()
