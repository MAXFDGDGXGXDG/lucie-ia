# Mettre Lucie en ligne

Objectif: avoir une adresse web publique pour que tout le monde puisse utiliser Lucie.

## Option simple: Render

1. Cree un compte sur Render.
2. Mets ce dossier dans un depot GitHub.
3. Dans Render, choisis **New Web Service**.
4. Connecte le depot.
5. Render detectera `render.yaml`.
6. Lance le deploiement.

Quand c'est fini, Render donne une adresse du style:

```text
https://lucie-ia.onrender.com
```

Tu peux partager cette adresse. Google pourra aussi l'indexer si la page reste publique.

## Commande de lancement

Si un hebergeur demande la commande de lancement:

```bash
python cloud_server.py
```

## Variables utiles

- `PORT`: donne par l'hebergeur automatiquement.
- `HOST`: mettre `0.0.0.0`.
- `LUCIE_DATA_DIR`: dossier ou Lucie garde sa memoire.
- `OPENAI_API_KEY`: optionnel, seulement si tu veux connecter un vrai modele OpenAI.

## Important

Ne mets pas ton PC personnel directement public sur internet. Utilise un hebergeur ou un tunnel temporaire pour tester.
