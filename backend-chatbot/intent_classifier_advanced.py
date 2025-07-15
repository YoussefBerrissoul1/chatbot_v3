"""
intent_classifier_advanced.py
-----------------------------
Classificateur d'intention RH avancé avec :
- Validation croisée et métriques de performance
- Hyperparameter tuning automatique
- Comparaison de plusieurs algorithmes
- Sauvegarde des métriques et visualisations
- Cache intelligent des modèles
"""

import joblib
import json
import os
import sys
import io
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

# Encodage UTF-8 pour la sortie
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Imports sklearn
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Imports pour visualisation
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Matplotlib/Seaborn non disponible - Visualisations désactivées")

class AdvancedIntentClassifier:
    """
    Classificateur d'intention RH avancé avec validation croisée et métriques
    """
    
    def __init__(self, data_path: str = "data/Nestle-HR-FAQ.json", 
                 model_dir: str = "model/", cache_enabled: bool = True):
        """
        Initialise le classificateur avancé
        
        Args:
            data_path (str): Chemin vers les données FAQ
            model_dir (str): Répertoire pour sauvegarder les modèles
            cache_enabled (bool): Active le cache des modèles
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.cache_enabled = cache_enabled
        
        # Chemins de sortie
        self.model_output = os.path.join(model_dir, "intent_classifier.pkl")
        self.metrics_output = os.path.join(model_dir, "classification_metrics.json")
        self.report_output = os.path.join(model_dir, "classification_report.txt")
        
        # Créer le répertoire de sortie
        os.makedirs(model_dir, exist_ok=True)
        
        # Algorithmes disponibles
        self.algorithms = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Paramètres pour GridSearch
        self.param_grids = {
            'logistic_regression': {
                'logisticregression__C': [0.1, 1, 10, 100],
                'logisticregression__solver': ['liblinear', 'lbfgs']
            },
            'naive_bayes': {
                'multinomialnb__alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'svm': {
                'svc__C': [0.1, 1, 10],
                'svc__gamma': ['scale', 'auto']
            },
            'random_forest': {
                'randomforestclassifier__n_estimators': [50, 100, 200],
                'randomforestclassifier__max_depth': [10, 20, None]
            }
        }
        
        # Stockage des résultats
        self.results = {}
        self.best_model = None
        self.best_algorithm = None
        self.label_encoder = LabelEncoder()

    def charger_donnees(self) -> Tuple[List[str], List[str]]:
        """
        Charge les données FAQ avec validation
        
        Returns:
            Tuple[List[str], List[str]]: Questions et thèmes correspondants
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si la structure des données est invalide
        """
        print("📊 Chargement des données FAQ...")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "faq" not in data:
                raise ValueError("Structure JSON invalide - clé 'faq' manquante")
            
            questions, themes = [], []
            theme_counts = {}
            
            for theme, items in data["faq"].items():
                theme_count = 0
                for item in items:
                    question = item.get("question", "").strip()
                    if question:  # Ignorer les questions vides
                        questions.append(question)
                        themes.append(theme)
                        theme_count += 1
                
                theme_counts[theme] = theme_count
            
            # Afficher les statistiques
            print(f"✅ {len(questions)} questions chargées")
            print(f"📋 {len(theme_counts)} thèmes détectés :")
            for theme, count in sorted(theme_counts.items()):
                print(f"   - {theme}: {count} questions")
            
            return questions, themes
            
        except FileNotFoundError:
            print(f"❌ Fichier non trouvé : {self.data_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Erreur JSON : {e}")
            raise
        except Exception as e:
            print(f"❌ Erreur inattendue : {e}")
            raise

    def verifier_cache_modele(self) -> bool:
        """
        Vérifie si un modèle en cache est disponible et valide
        
        Returns:
            bool: True si le cache est valide
        """
        if not self.cache_enabled:
            return False
        
        try:
            # Vérifier l'existence des fichiers
            if not os.path.exists(self.model_output):
                return False
            
            # Vérifier la date de modification
            model_time = os.path.getmtime(self.model_output)
            data_time = os.path.getmtime(self.data_path)
            
            if model_time < data_time:
                print("⚠️ Données plus récentes que le modèle - Réentraînement nécessaire")
                return False
            
            # Charger et valider le modèle
            model = joblib.load(self.model_output)
            if hasattr(model, 'predict'):
                print("✅ Modèle en cache valide trouvé")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la vérification du cache : {e}")
            return False

    def evaluer_modele(self, model, X_train, X_test, y_train, y_test, 
                      algorithm_name: str) -> Dict[str, Any]:
        """
        Évalue un modèle avec des métriques complètes
        
        Args:
            model: Modèle entraîné
            X_train, X_test: Données d'entraînement et de test
            y_train, y_test: Labels d'entraînement et de test
            algorithm_name (str): Nom de l'algorithme
            
        Returns:
            Dict[str, Any]: Métriques de performance
        """
        print(f"🔍 Évaluation du modèle {algorithm_name}...")
        
        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métriques de base
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Rapport de classification
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        
        results = {
            'algorithm': algorithm_name,
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'overfitting_score': float(train_accuracy - test_accuracy),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   ✅ Précision test : {test_accuracy:.3f}")
        print(f"   ✅ CV moyenne : {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        return results

    def optimiser_hyperparametres(self, algorithm_name: str, model, 
                                 X_train, y_train) -> Any:
        """
        Optimise les hyperparamètres avec GridSearchCV
        
        Args:
            algorithm_name (str): Nom de l'algorithme
            model: Modèle à optimiser
            X_train, y_train: Données d'entraînement
            
        Returns:
            Meilleur modèle optimisé
        """
        if algorithm_name not in self.param_grids:
            print(f"⚠️ Pas d'optimisation définie pour {algorithm_name}")
            return model
        
        print(f"🔧 Optimisation des hyperparamètres pour {algorithm_name}...")
        
        # Créer le pipeline
        pipeline = make_pipeline(TfidfVectorizer(), self.algorithms[algorithm_name])
        
        # GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            self.param_grids[algorithm_name],
            cv=3,  # Réduit pour accélérer
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"   ✅ Meilleur score : {grid_search.best_score_:.3f}")
        print(f"   ✅ Meilleurs paramètres : {grid_search.best_params_}")
        
        return grid_search.best_estimator_

    def comparer_algorithmes(self, questions: List[str], themes: List[str]) -> Dict[str, Any]:
        """
        Compare plusieurs algorithmes de classification
        
        Args:
            questions (List[str]): Questions FAQ
            themes (List[str]): Thèmes correspondants
            
        Returns:
            Dict[str, Any]: Résultats de comparaison
        """
        print("🏆 Comparaison des algorithmes de classification...")
        print("=" * 60)
        
        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(
            questions, themes, test_size=0.2, random_state=42, stratify=themes
        )
        
        results = {}
        
        for algorithm_name, algorithm in self.algorithms.items():
            try:
                # Créer le pipeline
                pipeline = make_pipeline(TfidfVectorizer(), algorithm)
                
                # Optimiser les hyperparamètres
                optimized_model = self.optimiser_hyperparametres(
                    algorithm_name, pipeline, X_train, y_train
                )
                
                # Entraîner le modèle optimisé
                optimized_model.fit(X_train, y_train)
                
                # Évaluer le modèle
                results[algorithm_name] = self.evaluer_modele(
                    optimized_model, X_train, X_test, y_train, y_test, algorithm_name
                )
                
                # Garder le modèle pour comparaison
                results[algorithm_name]['model'] = optimized_model
                
            except Exception as e:
                print(f"❌ Erreur avec {algorithm_name}: {e}")
                continue
        
        # Trouver le meilleur modèle
        if results:
            best_algorithm = max(results.keys(), 
                               key=lambda x: results[x]['cv_mean'])
            
            self.best_algorithm = best_algorithm
            self.best_model = results[best_algorithm]['model']
            
            print("=" * 60)
            print(f"🏆 Meilleur algorithme : {best_algorithm}")
            print(f"🎯 Score CV : {results[best_algorithm]['cv_mean']:.3f}")
            print("=" * 60)
        
        return results

    def sauvegarder_resultats(self, results: Dict[str, Any]):
        """
        Sauvegarde les résultats et métriques
        
        Args:
            results (Dict[str, Any]): Résultats de comparaison
        """
        print("💾 Sauvegarde des résultats...")
        
        # Préparer les données pour la sauvegarde (sans les modèles)
        save_results = {}
        for algo, data in results.items():
            save_results[algo] = {k: v for k, v in data.items() if k != 'model'}
        
        # Sauvegarder les métriques JSON
        with open(self.metrics_output, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder le meilleur modèle
        if self.best_model:
            joblib.dump(self.best_model, self.model_output)
            print(f"✅ Meilleur modèle sauvé : {self.model_output}")
        
        # Générer le rapport texte
        self.generer_rapport_texte(save_results)
        
        print(f"✅ Métriques sauvées : {self.metrics_output}")

    def generer_rapport_texte(self, results: Dict[str, Any]):
        """
        Génère un rapport texte détaillé
        
        Args:
            results (Dict[str, Any]): Résultats de comparaison
        """
        with open(self.report_output, 'w', encoding='utf-8') as f:
            f.write("RAPPORT DE CLASSIFICATION D'INTENTION RH\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Fichier de données : {self.data_path}\n\n")
            
            # Résumé des algorithmes
            f.write("COMPARAISON DES ALGORITHMES\n")
            f.write("-" * 30 + "\n")
            
            for algo, data in results.items():
                f.write(f"\n{algo.upper()}:\n")
                f.write(f"  Précision test      : {data['test_accuracy']:.3f}\n")
                f.write(f"  Validation croisée  : {data['cv_mean']:.3f} (±{data['cv_std']:.3f})\n")
                f.write(f"  Score overfitting   : {data['overfitting_score']:.3f}\n")
            
            # Meilleur modèle
            if self.best_algorithm:
                f.write(f"\nMEILLEUR MODÈLE : {self.best_algorithm}\n")
                f.write("-" * 30 + "\n")
                best_data = results[self.best_algorithm]
                f.write(f"Score final : {best_data['cv_mean']:.3f}\n")
        
        print(f"✅ Rapport généré : {self.report_output}")

    def generer_visualisations(self, results: Dict[str, Any]):
        """
        Génère des visualisations des résultats
        
        Args:
            results (Dict[str, Any]): Résultats de comparaison
        """
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Visualisations non disponibles - Modules manquants")
            return
        
        print("📈 Génération des visualisations...")
        
        # Créer le répertoire de visualisations
        vis_dir = os.path.join(self.model_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Comparaison des algorithmes
        algorithms = list(results.keys())
        cv_means = [results[algo]['cv_mean'] for algo in algorithms]
        test_accuracies = [results[algo]['test_accuracy'] for algo in algorithms]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique 1 : Scores de validation croisée
        bars1 = ax1.bar(algorithms, cv_means, color='skyblue', alpha=0.7)
        ax1.set_title('Scores de Validation Croisée')
        ax1.set_ylabel('Score CV')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars1, cv_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Graphique 2 : Précision sur test
        bars2 = ax2.bar(algorithms, test_accuracies, color='lightgreen', alpha=0.7)
        ax2.set_title('Précision sur Test')
        ax2.set_ylabel('Précision')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars2, test_accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Matrice de confusion pour le meilleur modèle
        if self.best_algorithm:
            best_conf_matrix = np.array(results[self.best_algorithm]['confusion_matrix'])
            
            # Récupérer les labels uniques
            questions, themes = self.charger_donnees()
            unique_themes = sorted(set(themes))
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=unique_themes, yticklabels=unique_themes)
            plt.title(f'Matrice de Confusion - {self.best_algorithm}')
            plt.xlabel('Prédiction')
            plt.ylabel('Réalité')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualisations sauvées dans : {vis_dir}")

    def entrainer_classifieur_complet(self):
        """
        Pipeline complet d'entraînement avec comparaison d'algorithmes
        """
        print("🚀 Démarrage de l'entraînement avancé du classifieur...")
        print("=" * 60)
        
        # Vérifier le cache
        if self.verifier_cache_modele():
            print("⚡ Utilisation du modèle en cache")
            self.best_model = joblib.load(self.model_output)
            return
        
        try:
            # Charger les données
            questions, themes = self.charger_donnees()
            
            # Comparer les algorithmes
            results = self.comparer_algorithmes(questions, themes)
            
            # Sauvegarder les résultats
            self.sauvegarder_resultats(results)
            
            # Générer les visualisations
            self.generer_visualisations(results)
            
            print("\n🎉 Entraînement terminé avec succès!")
            print(f"🏆 Meilleur modèle : {self.best_algorithm}")
            print(f"📊 Score final : {results[self.best_algorithm]['cv_mean']:.3f}")
            
        except Exception as e:
            print(f"❌ Erreur pendant l'entraînement : {e}")
            raise

    def predire_intention(self, question: str) -> Tuple[str, float]:
        """
        Prédit l'intention d'une question
        
        Args:
            question (str): Question à classifier
            
        Returns:
            Tuple[str, float]: (thème prédit, score de confiance)
        """
        if not self.best_model:
            # Charger le modèle si pas déjà fait
            if os.path.exists(self.model_output):
                self.best_model = joblib.load(self.model_output)
            else:
                raise ValueError("Aucun modèle entraîné disponible")
        
        # Prédire avec probabilité
        theme = self.best_model.predict([question])[0]
        
        # Obtenir la probabilité maximale comme score de confiance
        if hasattr(self.best_model, 'predict_proba'):
            proba = self.best_model.predict_proba([question])[0]
            confidence = float(max(proba))
        else:
            confidence = 0.8  # Valeur par défaut
        
        return theme, confidence


def main():
    """
    Fonction principale pour l'entraînement du classifieur avancé
    """
    print("🎯 Classifieur d'intention RH avancé")
    print("=" * 40)
    
    # Initialiser le classifieur
    classifier = AdvancedIntentClassifier()
    
    # Entraîner le modèle
    classifier.entrainer_classifieur_complet()
    
    # Test rapide
    print("\n🧪 Test rapide du modèle...")
    try:
        question_test = "Comment puis-je demander mes congés ?"
        theme, confidence = classifier.predire_intention(question_test)
        print(f"Question : {question_test}")
        print(f"Thème prédit : {theme}")
        print(f"Confiance : {confidence:.3f}")
    except Exception as e:
        print(f"⚠️ Erreur lors du test : {e}")


if __name__ == "__main__":
    main()