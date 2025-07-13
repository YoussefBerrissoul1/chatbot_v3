"""
chatbot_optimized.py 
=======================================
Chatbot RH Nestlé avec correction orthographique compatible
"""

import json
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict
import os
import sys

# Import des modules de machine learning
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Modules ML non disponibles - fonctionnalité limitée")

# Import du correcteur orthographique
try:
    from spellchecker import preprocess_question
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("⚠️ Correcteur orthographique non disponible")
    
    # Fonction fallback si le module n'est pas disponible
    def preprocess_question(texte: str) -> str:
        return texte.lower().strip()

class ChatbotRH:
    """Chatbot RH intelligent avec apprentissage automatique"""
    
    def __init__(self, fichier_donnees="donnees_rh.json", dossier_modeles="modeles_ml"):
        """Initialise le chatbot avec les données RH"""
        self.fichier_donnees = fichier_donnees
        self.dossier_modeles = dossier_modeles
        self.donnees = {}
        self.vectorizer = None
        self.classificateur = None
        self.questions_incomprises = []
        self.historique_conversations = []
        self.seuil_confiance = 0.3
        self.stats = {
            "questions_posees": 0,
            "reponses_trouvees": 0,
            "questions_incomprises": 0,
            "themes_populaires": defaultdict(int)
        }
        
        # Créer le dossier modèles s'il n'existe pas
        if not os.path.exists(self.dossier_modeles):
            os.makedirs(self.dossier_modeles)
        
        # Charger les données et modèles
        self.charger_donnees()
        if ML_AVAILABLE:
            self.charger_ou_entrainer_modeles()
    
    def charger_donnees(self):
        """Charge les données depuis le fichier JSON"""
        try:
            with open(self.fichier_donnees, 'r', encoding='utf-8') as f:
                self.donnees = json.load(f)
            print(f"✅ Données chargées : {len(self.donnees)} catégories")
        except FileNotFoundError:
            print(f"❌ Fichier {self.fichier_donnees} non trouvé")
            self.creer_donnees_exemple()
        except json.JSONDecodeError:
            print(f"❌ Erreur de format JSON dans {self.fichier_donnees}")
            self.creer_donnees_exemple()
    
    def creer_donnees_exemple(self):
        """Crée des données d'exemple si le fichier n'existe pas"""
        print("🔄 Création des données d'exemple...")
        self.donnees = {
            "congés": [
                {"question": "comment demander congés", "reponse": "Pour demander des congés, connectez-vous à votre espace RH et remplissez le formulaire de demande de congés."},
                {"question": "solde congés payés", "reponse": "Votre solde de congés payés est visible dans votre espace personnel RH."},
                {"question": "congés maladie", "reponse": "En cas de maladie, prévenez votre manager et envoyez votre arrêt de travail aux RH."}
            ],
            "salaire": [
                {"question": "bulletin paie", "reponse": "Votre bulletin de paie est disponible dans votre espace RH en format PDF."},
                {"question": "augmentation salaire", "reponse": "Les discussions sur les augmentations se font lors de votre entretien annuel."},
                {"question": "primes", "reponse": "Les primes sont calculées selon vos objectifs et la performance de l'entreprise."}
            ],
            "formation": [
                {"question": "formation professionnelle", "reponse": "Consultez le catalogue de formations disponible dans votre espace RH."},
                {"question": "cpf compte personnel formation", "reponse": "Vous pouvez utiliser votre CPF pour financer certaines formations."},
                {"question": "développement compétences", "reponse": "Parlez à votre manager de vos besoins en développement de compétences."}
            ],
            "télétravail": [
                {"question": "télétravail remote", "reponse": "Le télétravail est possible selon votre poste et avec accord de votre manager."},
                {"question": "horaires flexibles", "reponse": "Les horaires flexibles sont possibles selon votre fonction et équipe."},
                {"question": "travail distance", "reponse": "Le travail à distance suit des règles spécifiques définies par l'entreprise."}
            ]
        }
        
        # Sauvegarder les données d'exemple
        with open(self.fichier_donnees, 'w', encoding='utf-8') as f:
            json.dump(self.donnees, f, ensure_ascii=False, indent=2)
        
        print("✅ Données d'exemple créées")
    
    def charger_ou_entrainer_modeles(self):
        """Charge les modèles existants ou les entraîne"""
        chemin_vectorizer = os.path.join(self.dossier_modeles, "vectorizer.pkl")
        chemin_classificateur = os.path.join(self.dossier_modeles, "classificateur.pkl")
        
        if os.path.exists(chemin_vectorizer) and os.path.exists(chemin_classificateur):
            try:
                with open(chemin_vectorizer, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(chemin_classificateur, 'rb') as f:
                    self.classificateur = pickle.load(f)
                print("✅ Modèles chargés depuis le cache")
                return
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement : {e}")
        
        # Entraîner les modèles
        print("🔄 Entraînement des modèles...")
        self.entrainer_modeles()
        print("✅ Modèles entraînés et sauvegardés")
    
    def entrainer_modeles(self):
        """Entraîne les modèles de machine learning"""
        if not ML_AVAILABLE:
            print("⚠️ ML non disponible - utilisation des méthodes basiques")
            return
        
        # Préparer les données d'entraînement
        textes = []
        labels = []
        
        for theme, questions in self.donnees.items():
            for item in questions:
                textes.append(item["question"])
                labels.append(theme)
        
        if not textes:
            print("❌ Aucune donnée d'entraînement disponible")
            return
        
        # Vectorisation
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=None  # Pas de stop words pour le français
        )
        
        X = self.vectorizer.fit_transform(textes)
        y = labels
        
        # Entraînement du classificateur
        self.classificateur = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classificateur.fit(X, y)
        
        # Sauvegarde des modèles
        with open(os.path.join(self.dossier_modeles, "vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(self.dossier_modeles, "classificateur.pkl"), 'wb') as f:
            pickle.dump(self.classificateur, f)
    
    def detecter_theme(self, question: str) -> str:
        """Détecte le thème de la question"""
        if not ML_AVAILABLE or not self.vectorizer or not self.classificateur:
            # Méthode basique par mots-clés
            return self.detecter_theme_basique(question)
        
        try:
            # Vectorisation de la question
            question_vectorisee = self.vectorizer.transform([question])
            
            # Prédiction
            theme_predit = self.classificateur.predict(question_vectorisee)[0]
            
            # Mise à jour des stats
            self.stats["themes_populaires"][theme_predit] += 1
            
            return theme_predit
            
        except Exception as e:
            print(f"⚠️ Erreur de détection ML : {e}")
            return self.detecter_theme_basique(question)
    
    def detecter_theme_basique(self, question: str) -> str:
        """Détection basique par mots-clés"""
        question_lower = question.lower()
        
        # Mots-clés pour chaque thème
        themes_mots_cles = {
            "congés": ["congé", "congés", "vacances", "repos", "arrêt", "absence"],
            "salaire": ["salaire", "paie", "rémunération", "augmentation", "prime"],
            "formation": ["formation", "cours", "stage", "apprentissage", "cpf"],
            "télétravail": ["télétravail", "remote", "distance", "horaires", "flexible"],
            "général": []
        }
        
        # Compter les occurrences
        scores = {}
        for theme, mots_cles in themes_mots_cles.items():
            score = sum(1 for mot in mots_cles if mot in question_lower)
            if score > 0:
                scores[theme] = score
        
        # Retourner le thème avec le meilleur score
        if scores:
            theme_detecte = max(scores, key=scores.get)
            self.stats["themes_populaires"][theme_detecte] += 1
            return theme_detecte
        else:
            return "général"
    
    def trouver_meilleure_reponse(self, question: str) -> tuple:
        """Trouve la meilleure réponse pour une question"""
        meilleure_reponse = "Je ne sais pas répondre à cette question."
        meilleur_score = 0.0
        
        theme = self.detecter_theme(question)
        
        # Chercher dans le thème détecté d'abord
        if theme in self.donnees:
            for item in self.donnees[theme]:
                score = self.calculer_similarite(question, item["question"])
                if score > meilleur_score:
                    meilleur_score = score
                    meilleure_reponse = item["reponse"]
        
        # Si le score est trop faible, chercher dans tous les thèmes
        if meilleur_score < self.seuil_confiance:
            for theme_data in self.donnees.values():
                for item in theme_data:
                    score = self.calculer_similarite(question, item["question"])
                    if score > meilleur_score:
                        meilleur_score = score
                        meilleure_reponse = item["reponse"]
        
        return meilleure_reponse, meilleur_score
    
    def calculer_similarite(self, question1: str, question2: str) -> float:
        """Calcule la similarité entre deux questions"""
        if not ML_AVAILABLE or not self.vectorizer:
            # Méthode basique
            return self.calculer_similarite_basique(question1, question2)
        
        try:
            # Vectorisation des questions
            vecteurs = self.vectorizer.transform([question1, question2])
            
            # Calcul de la similarité cosinus
            similarite = cosine_similarity(vecteurs[0:1], vecteurs[1:2])[0][0]
            
            return similarite
            
        except Exception as e:
            print(f"⚠️ Erreur de calcul de similarité : {e}")
            return self.calculer_similarite_basique(question1, question2)
    
    def calculer_similarite_basique(self, question1: str, question2: str) -> float:
        """Calcule la similarité basique entre deux questions"""
        mots1 = set(question1.lower().split())
        mots2 = set(question2.lower().split())
        
        if not mots1 and not mots2:
            return 1.0
        if not mots1 or not mots2:
            return 0.0
        
        # Calcul de l'intersection et de l'union
        intersection = len(mots1.intersection(mots2))
        union = len(mots1.union(mots2))
        
        return intersection / union if union > 0 else 0.0
    
    def generer_reponse(self, question: str) -> str:
        """Génère une réponse à la question de l'utilisateur"""
        self.stats["questions_posees"] += 1
        
        if not question.strip():
            return "⚠️ Veuillez poser une question."
        
        # 🔥 CORRECTION : Utiliser le correcteur orthographique
        question_originale = question
        question_corrigee = preprocess_question(question)
        
        # Afficher la correction seulement si différente
        if question_originale.strip() != question_corrigee.strip():
            print(f"📝 Correction: '{question_originale}' → '{question_corrigee}'")
        
        # Détecter le thème (utiliser la version corrigée)
        theme = self.detecter_theme(question_corrigee)
        
        # Trouver la meilleure réponse (utiliser la version corrigée)
        reponse, score = self.trouver_meilleure_reponse(question_corrigee)
        
        # Enregistrer dans l'historique
        self.historique_conversations.append({
            "timestamp": datetime.now().isoformat(),
            "question_originale": question_originale,
            "question_corrigee": question_corrigee,
            "theme": theme,
            "reponse": reponse,
            "score": score
        })
        
        # Afficher des informations de debug
        print(f"📁 Thème détecté: {theme.replace('_', ' ').title()}")
        print(f"🎯 Score de confiance: {score:.2f}")
        
        if score >= self.seuil_confiance:
            self.stats["reponses_trouvees"] += 1
            return f"✅ {reponse}"
        else:
            self.stats["questions_incomprises"] += 1
            self.enregistrer_question_incomprise(question_corrigee, theme)
            return (f"❌ Je n'ai pas bien compris votre question (score: {score:.2f}).\n"
                   f"Voici quand même ma meilleure suggestion :\n{reponse}\n\n"
                   f"💡 Essayez de reformuler votre question ou d'être plus spécifique.")
    
    def enregistrer_question_incomprise(self, question: str, theme: str):
        """Enregistre une question non comprise pour amélioration future"""
        self.questions_incomprises.append({
            "question": question,
            "theme_suggere": theme,
            "timestamp": datetime.now().isoformat()
        })
        
        # Sauvegarder dans un fichier
        try:
            with open("questions_incomprises.json", "w", encoding="utf-8") as f:
                json.dump(self.questions_incomprises, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde questions incomprises : {e}")
    
    def afficher_stats(self):
        """Affiche les statistiques du chatbot"""
        print("\n" + "="*50)
        print("📊 STATISTIQUES DU CHATBOT")
        print("="*50)
        print(f"Questions posées: {self.stats['questions_posees']}")
        print(f"Réponses trouvées: {self.stats['reponses_trouvees']}")
        print(f"Questions incomprises: {self.stats['questions_incomprises']}")
        
        if self.stats['questions_posees'] > 0:
            taux_reussite = (self.stats['reponses_trouvees'] / self.stats['questions_posees']) * 100
            print(f"Taux de réussite: {taux_reussite:.1f}%")
        
        print(f"Seuil de confiance: {self.seuil_confiance}")
        print(f"Correcteur orthographique: {'✅' if SPELLCHECKER_AVAILABLE else '❌'}")
        print(f"Machine Learning: {'✅' if ML_AVAILABLE else '❌'}")
        
        if self.stats['themes_populaires']:
            print("\nThèmes populaires:")
            for theme, count in sorted(self.stats['themes_populaires'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {theme}: {count} questions")
        
        print("="*50)
    
    def tester_modele(self):
        """Teste le modèle avec des questions d'exemple"""
        questions_test = [
            "CONGES",
            "Puis-je travailler",
            "Comment demander des congés?",
            "Quel est mon salaire?",
            "Formation disponible",
            "Télétravail possible?",
            "Coment faire demande congé",
            "Problème avec mon manageur"
        ]
        
        print("\n" + "="*50)
        print("🧪 TEST DU MODÈLE")
        print("="*50)
        
        for i, question in enumerate(questions_test, 1):
            print(f"\n{i}. Question: {question}")
            reponse = self.generer_reponse(question)
            print(f"   Réponse: {reponse}")
            print("-" * 40)
    
    def ajuster_seuil_confiance(self, nouveau_seuil: float):
        """Ajuste le seuil de confiance"""
        if 0.0 <= nouveau_seuil <= 1.0:
            self.seuil_confiance = nouveau_seuil
            print(f"✅ Seuil de confiance ajusté à {nouveau_seuil}")
        else:
            print("❌ Le seuil doit être entre 0.0 et 1.0")
    
    def sauvegarder_historique(self):
        """Sauvegarde l'historique des conversations"""
        try:
            with open("historique_conversations.json", "w", encoding="utf-8") as f:
                json.dump(self.historique_conversations, f, ensure_ascii=False, indent=2)
            print("✅ Historique sauvegardé")
        except Exception as e:
            print(f"❌ Erreur sauvegarde historique : {e}")
    
    def charger_historique(self):
        """Charge l'historique des conversations"""
        try:
            with open("historique_conversations.json", "r", encoding="utf-8") as f:
                self.historique_conversations = json.load(f)
            print(f"✅ Historique chargé : {len(self.historique_conversations)} conversations")
        except FileNotFoundError:
            print("📝 Aucun historique existant")
        except Exception as e:
            print(f"❌ Erreur chargement historique : {e}")

def main():
    """Fonction principale du chatbot"""
    print("🚀 Initialisation du Chatbot RH Nestlé...")
    
    # Créer l'instance du chatbot
    chatbot = ChatbotRH()
    
    # Charger l'historique
    chatbot.charger_historique()
    
    print("✅ Chatbot initialisé avec les modèles sauvegardés")
    print("\n💬 Chatbot RH Nestlé - Version Optimisée")
    print("Tapez 'exit', 'quit' ou 'test' pour les options spéciales")
    
    try:
        while True:
            # Demander une question à l'utilisateur
            question = input("\nVous: ").strip()
            
            # Commandes spéciales
            if question.lower() in ['exit', 'quit', 'quitter']:
                print("👋 Au revoir ! Sauvegarde en cours...")
                chatbot.sauvegarder_historique()
                chatbot.afficher_stats()
                break
            
            elif question.lower() == 'test':
                chatbot.tester_modele()
                continue
            
            elif question.lower() == 'stats':
                chatbot.afficher_stats()
                continue
            
            elif question.lower().startswith('seuil'):
                try:
                    # Commande: seuil 0.4
                    parts = question.split()
                    if len(parts) == 2:
                        nouveau_seuil = float(parts[1])
                        chatbot.ajuster_seuil_confiance(nouveau_seuil)
                    else:
                        print("Usage: seuil 0.4")
                except ValueError:
                    print("❌ Valeur invalide pour le seuil")
                continue
            
            elif question.lower() == 'help':
                print("\n🔧 Commandes disponibles:")
                print("  - exit/quit : Quitter le chatbot")
                print("  - test : Tester le modèle")
                print("  - stats : Afficher les statistiques")
                print("  - seuil X : Ajuster le seuil de confiance (ex: seuil 0.4)")
                print("  - help : Afficher cette aide")
                continue
            
            # Traiter la question normale
            if question:
                reponse = chatbot.generer_reponse(question)
                print(f"\n🤖 Chatbot: {reponse}")
            else:
                print("⚠️ Veuillez poser une question.")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt du chatbot...")
        chatbot.sauvegarder_historique()
        chatbot.afficher_stats()
    
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        chatbot.sauvegarder_historique()
        chatbot.afficher_stats()

if __name__ == "__main__":
    main()