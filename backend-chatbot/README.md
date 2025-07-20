# Backend Chatbot – Déploiement et Utilisation

## Déploiement recommandé

Ce backend Python n'est pas compatible avec Netlify (qui ne supporte que Node.js/Go pour les fonctions serverless). Pour le déployer facilement :

- **Render.com** (recommandé)
- **Railway.app**
- **Heroku**
- **AWS Lambda (via Zappa ou Serverless Framework)**

### Étapes générales (exemple Render)
1. Crée un nouveau service web Python sur Render
2. Connecte ce dossier `backend-chatbot/` (ou ton repo Git)
3. Commande de démarrage :
   ```bash
   python api.py
   ```
4. Ajoute les variables d'environnement nécessaires (si besoin)
5. Récupère l'URL publique de l'API
6. Configure le frontend (Netlify) pour pointer vers cette URL (dans le code ou via une variable d'environnement)

## Lancement local
```bash
cd backend-chatbot
pip install -r requirements.txt
python api.py
```

## Sécurité
- Ne jamais exposer ce backend sans protection si tu ajoutes des endpoints sensibles.
- Utilise CORS pour limiter les domaines autorisés.

## Contact
Pour toute question, contacte l'équipe technique RH Nestlé. 