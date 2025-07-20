from flask import Flask, request, jsonify
from chatbot_optimized import ChatbotRHOptimise
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app)  # Autorise les requêtes du frontend

# Initialisation du logger robuste
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('log/api.log'),
        logging.StreamHandler()
    ]
)

# Initialisation unique du chatbot (chargement optimisé)
chatbot = None

def get_chatbot():
    global chatbot
    if chatbot is None:
        chatbot = ChatbotRHOptimise()
        chatbot.initialiser()
    return chatbot

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logging.warning('Requête invalide reçue: %s', data)
            return jsonify({'error': 'Requête invalide.'}), 400
        user_message = data['message']
        if not isinstance(user_message, str) or len(user_message.strip()) == 0 or len(user_message) > 1000:
            logging.warning('Message utilisateur invalide: %s', user_message)
            return jsonify({'error': 'Message utilisateur invalide.'}), 400
        bot = get_chatbot()
        bot_response = bot.generer_reponse(user_message)
        logging.info('Question: %s | Réponse: %s', user_message, bot_response)
        return jsonify({'response': bot_response})
    except Exception as e:
        logging.error('Erreur interne: %s', e, exc_info=True)
        return jsonify({'error': "Erreur interne du serveur. Veuillez réessayer plus tard."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 