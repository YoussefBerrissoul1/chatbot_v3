from flask import Flask, request, jsonify
from chatbot_optimized import ChatbotRHOptimise
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Autorise les requêtes du frontend

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
    data = request.get_json()
    user_message = data.get('message', '')
    bot = get_chatbot()
    bot_response = bot.generer_reponse(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 