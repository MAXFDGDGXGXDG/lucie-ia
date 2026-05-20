# IA Apprend

Petit projet d'IA conversationnelle qui utilise un vrai modele OpenAI quand `OPENAI_API_KEY` est defini, et passe en mode local sinon.

## Ce que fait ce projet

- repond en francais par defaut
- garde une memoire des echanges recents
- apprend des faits avec `/teach question | reponse`
- charge un petit jeu de depart dans [qa_seed.json](../qa_seed.json)
- affiche une interface web simple avec la zone de saisie au centre

## Lancer le projet

Interface web par defaut:

```powershell
python .\ia_apprend\main.py
```

Mode console:

```powershell
python .\ia_apprend\main.py --mode cli
```

## Configuration du vrai modele

Definis la variable d'environnement `OPENAI_API_KEY` pour activer le modele distant.

Optionnellement, tu peux changer le modele avec `IA_MODEL`. Par defaut, le projet utilise `gpt-5.5`.

## Commandes utiles

- `/teach question | reponse` : enseigne un nouveau fait
- `/list` : affiche les connaissances enregistrees
- `/help` : affiche l'aide
- `/exit` : quitte le programme
