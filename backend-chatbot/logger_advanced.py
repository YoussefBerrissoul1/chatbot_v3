"""
logger_advanced.py
------------------
Logger avancé pour le chatbot RH avec métriques et analytics
"""

import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

class AdvancedQuestionLogger:
    """
    Logger avancé pour enregistrer les questions et générer des métriques
    """

    def __init__(self, log_path: str = "logs/chatbot_logs.json"):
        self.log_path = log_path
        self.csv_path = log_path.replace('.json', '.csv')
        self.metrics_path = log_path.replace('.json', '_metrics.json')
        
        # Créer les répertoires nécessaires
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Initialiser les fichiers
        self._init_files()

    def _init_files(self):
        """Initialise les fichiers de log s'ils n'existent pas"""
        if not os.path.isfile(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump({
                    "sessions": [],
                    "questions": [],
                    "metrics": {
                        "total_questions": 0,
                        "answered_questions": 0,
                        "unanswered_questions": 0,
                        "themes_distribution": {},
                        "confidence_scores": []
                    }
                }, f, indent=2, ensure_ascii=False)

    def log_question(self, question: str, theme_detected: str = None, 
                    confidence: float = 0.0, response_found: bool = False,
                    response_time: float = 0.0, user_satisfied: bool = None):
        """
        Enregistre une question avec métadonnées complètes
        
        Args:
            question (str): Question posée par l'utilisateur
            theme_detected (str): Thème RH détecté
            confidence (float): Score de confiance du modèle
            response_found (bool): Réponse trouvée ou non
            response_time (float): Temps de réponse en secondes
            user_satisfied (bool): Satisfaction utilisateur (optionnel)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Charger les données existantes
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"sessions": [], "questions": [], "metrics": {}}

        # Créer l'entrée de log
        log_entry = {
            "timestamp": timestamp,
            "question": question,
            "theme_detected": theme_detected,
            "confidence": confidence,
            "response_found": response_found,
            "response_time": response_time,
            "user_satisfied": user_satisfied,
            "question_length": len(question.split()),
            "question_type": self._classify_question_type(question)
        }

        # Ajouter à la liste des questions
        data["questions"].append(log_entry)
        
        # Mettre à jour les métriques
        self._update_metrics(data, log_entry)
        
        # Sauvegarder le JSON
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder en CSV pour analyse
        self._save_to_csv(log_entry)
        
        print(f"📝 Question enregistrée avec métriques")

    def _classify_question_type(self, question: str) -> str:
        """Classifie le type de question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['quoi', 'que', 'qu\'est', 'quelle', 'quel']):
            return "definition"
        elif any(word in question_lower for word in ['comment', 'de quelle manière']):
            return "procedure"
        elif any(word in question_lower for word in ['pourquoi', 'pour quelle raison']):
            return "explication"
        elif any(word in question_lower for word in ['où', 'quand', 'à quelle heure']):
            return "information"
        elif '?' in question:
            return "question"
        else:
            return "affirmation"

    def _update_metrics(self, data: Dict, log_entry: Dict):
        """Met à jour les métriques globales"""
        metrics = data.get("metrics", {})
        
        # Compteurs de base
        metrics["total_questions"] = metrics.get("total_questions", 0) + 1
        
        if log_entry["response_found"]:
            metrics["answered_questions"] = metrics.get("answered_questions", 0) + 1
        else:
            metrics["unanswered_questions"] = metrics.get("unanswered_questions", 0) + 1
        
        # Distribution des thèmes
        if "themes_distribution" not in metrics:
            metrics["themes_distribution"] = {}
        
        theme = log_entry["theme_detected"] or "unknown"
        metrics["themes_distribution"][theme] = metrics["themes_distribution"].get(theme, 0) + 1
        
        # Scores de confiance
        if "confidence_scores" not in metrics:
            metrics["confidence_scores"] = []
        metrics["confidence_scores"].append(log_entry["confidence"])
        
        # Temps de réponse
        if "response_times" not in metrics:
            metrics["response_times"] = []
        metrics["response_times"].append(log_entry["response_time"])
        
        # Satisfaction utilisateur
        if log_entry["user_satisfied"] is not None:
            if "satisfaction_scores" not in metrics:
                metrics["satisfaction_scores"] = []
            metrics["satisfaction_scores"].append(log_entry["user_satisfied"])
        
        data["metrics"] = metrics

    def _save_to_csv(self, log_entry: Dict):
        """Sauvegarde les données en CSV pour analyse Excel"""
        file_exists = os.path.isfile(self.csv_path)
        
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(log_entry)

    def get_performance_metrics(self) -> Dict:
        """Retourne les métriques de performance"""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            metrics = data.get("metrics", {})
            
            # Calculer les métriques dérivées
            total = metrics.get("total_questions", 0)
            answered = metrics.get("answered_questions", 0)
            
            if total > 0:
                success_rate = (answered / total) * 100
                
                # Confiance moyenne
                confidence_scores = metrics.get("confidence_scores", [])
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                
                # Temps de réponse moyen
                response_times = metrics.get("response_times", [])
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                
                # Satisfaction moyenne
                satisfaction_scores = metrics.get("satisfaction_scores", [])
                avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else None
                
                return {
                    "total_questions": total,
                    "success_rate": round(success_rate, 2),
                    "avg_confidence": round(avg_confidence, 3),
                    "avg_response_time": round(avg_response_time, 3),
                    "avg_satisfaction": round(avg_satisfaction, 2) if avg_satisfaction else None,
                    "themes_distribution": metrics.get("themes_distribution", {}),
                    "most_common_theme": max(metrics.get("themes_distribution", {}).items(), key=lambda x: x[1])[0] if metrics.get("themes_distribution") else None
                }
            
            return {"total_questions": 0, "success_rate": 0}
            
        except Exception as e:
            print(f"⚠️ Erreur lors du calcul des métriques : {e}")
            return {}

    def generate_report(self) -> str:
        """Génère un rapport détaillé des performances"""
        metrics = self.get_performance_metrics()
        
        if not metrics:
            return "Aucune donnée disponible pour le rapport."
        
        report = f"""
📊 RAPPORT DE PERFORMANCE CHATBOT RH
====================================

🎯 Métriques générales :
- Total questions       : {metrics.get('total_questions', 0)}
- Taux de succès        : {metrics.get('success_rate', 0)}%
- Confiance moyenne     : {metrics.get('avg_confidence', 0)}
- Temps réponse moyen   : {metrics.get('avg_response_time', 0)}s

🔍 Analyse par thème :
"""
        
        themes = metrics.get('themes_distribution', {})
        for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / metrics['total_questions']) * 100
            report += f"- {theme}: {count} questions ({percentage:.1f}%)\n"
        
        if metrics.get('avg_satisfaction'):
            report += f"\n😊 Satisfaction moyenne : {metrics['avg_satisfaction']}/5"
        
        report += f"\n\n📈 Thème le plus fréquent : {metrics.get('most_common_theme', 'N/A')}"
        
        return report

    def export_analytics(self, output_path: str = "logs/analytics_report.json"):
        """Exporte une analyse complète en JSON"""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            analytics = {
                "generated_at": datetime.now().isoformat(),
                "performance_metrics": self.get_performance_metrics(),
                "raw_data_summary": {
                    "total_sessions": len(data.get("sessions", [])),
                    "total_questions": len(data.get("questions", [])),
                    "data_period": {
                        "first_question": data["questions"][0]["timestamp"] if data.get("questions") else None,
                        "last_question": data["questions"][-1]["timestamp"] if data.get("questions") else None
                    }
                },
                "recommendations": self._generate_recommendations(data)
            }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analytics, f, indent=2, ensure_ascii=False)
            
            print(f"📈 Rapport d'analyse exporté : {output_path}")
            
        except Exception as e:
            print(f"⚠️ Erreur lors de l'export : {e}")

    def _generate_recommendations(self, data: Dict) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        metrics = data.get("metrics", {})
        
        # Analyser le taux de succès
        total = metrics.get("total_questions", 0)
        answered = metrics.get("answered_questions", 0)
        
        if total > 0:
            success_rate = (answered / total) * 100
            
            if success_rate < 70:
                recommendations.append("Taux de succès faible - Améliorer la base de connaissances")
            
            # Analyser la confiance moyenne
            confidence_scores = metrics.get("confidence_scores", [])
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                if avg_confidence < 0.7:
                    recommendations.append("Confiance faible - Réentraîner le modèle")
            
            # Analyser les thèmes
            themes = metrics.get("themes_distribution", {})
            if themes:
                max_theme = max(themes.items(), key=lambda x: x[1])
                if max_theme[1] / total > 0.5:
                    recommendations.append(f"Thème dominant ({max_theme[0]}) - Diversifier la base de connaissances")
        
        return recommendations if recommendations else ["Performance satisfaisante - Continuer le monitoring"]