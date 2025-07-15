"""
Version sécurisée du chatbot avec gestion des variables d'environnement
"""

import json
import os
import pickle
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import unicodedata
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Import conditionnel du correcteur orthographique
try:
    from spellchecker import preprocess_question
except ImportError:
    def preprocess_question(question: str) -> str:
        """Fallback si le correcteur orthographique n'est pas disponible"""
        return question.strip()

def remove_accents(text: str) -> str:
    """Supprime les accents d'un texte pour éviter les problèmes d'encodage"""
    if not text:
        return text
    # Normaliser et supprimer les accents
    nfkd_form = unicodedata.normalize('NFD', text)
    ascii_text = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    return ascii_text

def safe_encode_text(text: str) -> str:
    """Encode un texte de manière sûre pour l'API"""
    if not text:
        return text
    
    # Remplacer les caractères problématiques
    replacements = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'î': 'i', 'ï': 'i',
        'ô': 'o', 'ö': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c',
        'ñ': 'n'
    }
    
    result = text
    for accent, replacement in replacements.items():
        result = result.replace(accent, replacement)
        result = result.replace(accent.upper(), replacement.upper())
    
    # Supprimer tout caractère non-ASCII restant
    result = result.encode('ascii', errors='ignore').decode('ascii')
    return result

class ChatbotRHOptimise:
    """
    Chatbot RH optimisé avec gestion sécurisée des variables d'environnement
    """
    
    def __init__(self, data_path: str = None):
        # Configuration depuis les variables d'environnement
        self.data_path = data_path or os.getenv("DATA_PATH", "data/Nestle-HR-FAQ.json")
        self.seuil_confiance = float(os.getenv("CONFIDENCE_THRESHOLD", "1.0"))
        
        # Configuration du vectorizer depuis l'environnement
        self.vectorizer = TfidfVectorizer(
            max_features=int(os.getenv("TFIDF_MAX_FEATURES", "5000")),
            ngram_range=(1, int(os.getenv("TFIDF_NGRAM_RANGE", "2"))),
            stop_words=None,
            lowercase=True,
            min_df=int(os.getenv("TFIDF_MIN_DF", "1")),
            max_df=float(os.getenv("TFIDF_MAX_DF", "0.95"))
        )
        
        # Configuration du classifier depuis l'environnement
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=int(os.getenv("CLASSIFIER_MAX_FEATURES", "3000")), 
                ngram_range=(1, int(os.getenv("CLASSIFIER_NGRAM_RANGE", "2")))
            )),
            ('nb', MultinomialNB(alpha=float(os.getenv("NAIVE_BAYES_ALPHA", "0.1"))))
        ])
        
        # Données
        self.questions = []
        self.reponses = []
        self.themes = []
        self.tfidf_matrix = None
        self.theme_data = {}
        
        # Statistiques
        self.stats = {
            "questions_posees": 0,
            "reponses_trouvees": 0,
            "questions_incomprises": 0
        }
        
        # Configuration OpenAI sécurisée
        self._initialiser_client_openai()
    
    def _initialiser_client_openai(self):
        """Initialise le client OpenAI avec les variables d'environnement"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            
            if not api_key:
                print("⚠️ Clé API OpenAI manquante dans le fichier .env")
                self.client = None
                return
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            print("✅ Client OpenAI initialisé avec succès")
            
        except Exception as e:
            print(f"⚠️ Erreur lors de l'initialisation du client OpenAI: {e}")
            self.client = None
    
    def nettoyer_texte(self, texte: str) -> str:
        """Nettoie, corrige et normalise le texte"""
        if not texte:
            return ""
        try:
            texte_corrige = preprocess_question(texte)
        except Exception:
            texte_corrige = texte.strip()
        
        texte_corrige = texte_corrige.lower()
        texte_corrige = re.sub(r'[^\w\s]', ' ', texte_corrige)
        texte_corrige = re.sub(r'\s+', ' ', texte_corrige)
        return texte_corrige.strip()

    def charger_donnees(self) -> None:
        """Charge les données FAQ depuis le fichier JSON"""
        print("📂 Chargement des données FAQ...")
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Fichier de données non trouvé: {self.data_path}")
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for theme, items in data["faq"].items():
                self.theme_data[theme] = []
                for item in items:
                    question = self.nettoyer_texte(item["question"])
                    reponse = item["response"]
                    self.questions.append(question)
                    self.reponses.append(reponse)
                    self.themes.append(theme)
                    self.theme_data[theme].append({
                        "question": question,
                        "response": reponse
                    })
            
            print(f"✅ {len(self.questions)} questions chargées dans {len(self.theme_data)} thèmes")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            raise
    
    def entrainer_modeles(self) -> None:
        """Entraîne les modèles TF-IDF et de classification"""
        print("🎯 Entraînement des modèles...")
        if self.questions:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)
            self.classifier.fit(self.questions, self.themes)
            print("✅ Modèles entraînés avec succès")
        else:
            print("⚠️ Dataset vide, utilisation de l'API par défaut")
    
    def detecter_theme(self, question: str) -> str:
        """Détecte le thème de la question"""
        try:
            question_nettoyee = self.nettoyer_texte(question)
            theme_predit = self.classifier.predict([question_nettoyee])[0]
            return theme_predit
        except Exception:
            return "inconnu"
    
    def trouver_meilleure_reponse(self, question: str) -> Tuple[str, float]:
        """Trouve la meilleure réponse à une question donnée"""
        if not self.questions:
            return "Aucune donnée disponible", 0.0
        question_nettoyee = self.nettoyer_texte(question)
        question_vector = self.vectorizer.transform([question_nettoyee])
        similarites = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        meilleur_index = np.argmax(similarites)
        meilleur_score = similarites[meilleur_index]
        return self.reponses[meilleur_index], meilleur_score
    
    def generer_reponse(self, question: str) -> str:
        """Génère une réponse à la question de l'utilisateur avec fallback vers l'API"""
        self.stats["questions_posees"] += 1
        
        if not question.strip():
            return "⚠️ Veuillez poser une question."
        
        question_originale = question
        try:
            question_corrigee = preprocess_question(question)
        except Exception:
            question_corrigee = question.strip()
        
        if question_originale != question_corrigee:
            print(f"📝 Correction: '{question_originale}' → '{question_corrigee}'")
        
        theme = self.detecter_theme(question_corrigee)
        reponse, score = self.trouver_meilleure_reponse(question_corrigee)
        
        print(f"📁 Thème détecté: {theme.replace('_', ' ').title()}")
        print(f"🎯 Score de confiance: {score:.2f}")
        
        if score >= self.seuil_confiance and self.questions:
            self.stats["reponses_trouvees"] += 1
            return f"✅ {reponse}"
        else:
            # Appel à l'API OpenRouter avec encodage ASCII-safe
            if not self.client:
                self.stats["questions_incomprises"] += 1
                return "❌ Service d'IA indisponible. Veuillez vérifier votre configuration."
            
            try:
                # Convertir la question en ASCII-safe
                question_ascii = safe_encode_text(question_corrigee)
                
                # Configuration du modèle depuis l'environnement
                model_name = os.getenv("OPENAI_MODEL", "openai/gpt-4o")
                max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "300"))
                temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
                
                # Message système configurable
                system_message = os.getenv(
                    "SYSTEM_MESSAGE", 
                    "You are an HR assistant for Nestle. Answer in French in a professional and helpful manner. Limit your response to 200 words maximum."
                )
                
                if os.getenv("DEBUG", "false").lower() == "true":
                    print(f"DEBUG: Question ASCII-safe: {repr(question_ascii)}")
                    print(f"DEBUG: Modèle utilisé: {model_name}")
                
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": question_ascii
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if os.getenv("DEBUG", "false").lower() == "true":
                    print("DEBUG: Réponse reçue de l'API")
                
                ai_answer = completion.choices[0].message.content
                if ai_answer:
                    ai_answer = ai_answer.strip()
                    return f"🤖 {ai_answer}"
                else:
                    self.stats["questions_incomprises"] += 1
                    return "❌ Réponse vide de l'IA. Veuillez reformuler votre question."
                    
            except Exception as e:
                if os.getenv("DEBUG", "false").lower() == "true":
                    print(f"❌ Erreur API détaillée: {str(e)}")
                else:
                    print(f"❌ Erreur API: {str(e)}")
                self.stats["questions_incomprises"] += 1
                return "❌ Service temporairement indisponible. Veuillez réessayer plus tard."
    
    def sauvegarder_modeles(self) -> None:
        """Sauvegarde les modèles entraînés"""
        model_dir = os.getenv("MODEL_DIR", "model")
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            with open(f"{model_dir}/vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(f"{model_dir}/classifier.pkl", 'wb') as f:
                pickle.dump(self.classifier, f)
            with open(f"{model_dir}/chatbot_data.pkl", 'wb') as f:
                pickle.dump({
                    'questions': self.questions,
                    'reponses': self.reponses,
                    'themes': self.themes,
                    'tfidf_matrix': self.tfidf_matrix
                }, f)
            print("💾 Modèles sauvegardés avec succès")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
    
    def charger_modeles(self) -> bool:
        """Charge les modèles sauvegardés"""
        model_dir = os.getenv("MODEL_DIR", "model")
        try:
            with open(f"{model_dir}/vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(f"{model_dir}/classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
            with open(f"{model_dir}/chatbot_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.questions = data['questions']
                self.reponses = data['reponses']
                self.themes = data['themes']
                self.tfidf_matrix = data['tfidf_matrix']
            print("✅ Modèles chargés depuis le cache")
            return True
        except Exception as e:
            if os.getenv("DEBUG", "false").lower() == "true":
                print(f"⚠️ Impossible de charger les modèles depuis le cache : {e}")
            return False
    
    def initialiser(self, force_retrain: bool = False) -> None:
        """Initialise le chatbot"""
        print("🚀 Initialisation du Chatbot RH Nestlé...")
        
        # Vérifier les variables d'environnement critiques
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ ATTENTION: Clé API OpenAI manquante dans le fichier .env")
        
        if not force_retrain and self.charger_modeles():
            print("✅ Chatbot initialisé avec les modèles sauvegardés")
            return
        
        self.charger_donnees()
        self.entrainer_modeles()
        self.sauvegarder_modeles()
        print("✅ Chatbot initialisé et prêt à répondre!")
    
    def afficher_statistiques(self) -> None:
        """Affiche les statistiques de la session"""
        print("\n📊 STATISTIQUES DE SESSION:")
        print(f"  Questions posées: {self.stats['questions_posees']}")
        print(f"  Réponses trouvées: {self.stats['reponses_trouvees']}")
        print(f"  Questions incomprises: {self.stats['questions_incomprises']}")
        if self.stats['questions_posees'] > 0:
            taux_reussite = (self.stats['reponses_trouvees'] / self.stats['questions_posees']) * 100
            print(f"  Taux de réussite: {taux_reussite:.1f}%")
    
    def executer(self) -> None:
        """Lance la boucle interactive du chatbot"""
        print("\n💬 Chatbot RH Nestlé - Version Sécurisée")
        print("   Tapez 'exit', 'quit' pour quitter\n")
        
        while True:
            try:
                question = input("Vous: ").strip()
                if question.lower() in ['exit', 'quit', 'sortir']:
                    print("👋 Merci d'avoir utilisé le chatbot RH!")
                    self.afficher_statistiques()
                    break
                if not question:
                    print("⚠️ Veuillez poser une question.")
                    continue
                reponse = self.generer_reponse(question)
                print(f"Bot: {reponse}\n")
            except KeyboardInterrupt:
                print("\n👋 Arrêt du chatbot.")
                self.afficher_statistiques()
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                continue

def main():
    """Point d'entrée principal"""
    try:
        # Vérifier si le fichier .env existe
        if not os.path.exists('.env'):
            print("❌ Fichier .env manquant. Veuillez le créer avec les variables requises.")
            return
        
        chatbot = ChatbotRHOptimise()
        chatbot.initialiser()
        chatbot.executer()
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")

if __name__ == "__main__":
    main()

