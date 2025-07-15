"""
run_all.py
----------
Pipeline complet pour entraîner le chatbot RH Nestlé

✅ Étapes automatisées :
   1. Augmentation des données (synonymes)
   2. Entraînement du classificateur d'intention
   3. Entraînement du chatbot optimisé

Auteur : Toi 😎
"""

import subprocess
import sys
import os

def run_script(script_name):
    print("\n" + "="*60)
    print(f"🚀 Exécution du script : {script_name}")
    print("="*60)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )    
    print(result.stdout)
    if result.stderr:
        print("⚠️ ERREUR :")
        print(result.stderr)
    print("="*60 + "\n")

def main():
    print("\n✨ Bienvenue dans le pipeline automatisé du Chatbot RH Nestlé ✨\n")

    # Vérification des répertoires nécessaires
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    # 1️⃣ Augmentation des données
    print("📊 Étape 1 : Augmentation des données...")
    run_script("augmenter_donnees.py")

    # 2️⃣ Entraînement du classificateur d'intention
    print("🎯 Étape 2 : Entraînement du classificateur d'intention...")
    run_script("intent_classifier_advanced.py")

    # 3️⃣ Entraînement du chatbot optimisé (avec correction orthographique)
    print("🤖 Étape 3 : Entraînement du chatbot optimisé...")
    # Note: chatbot_optimized.py fait maintenant l'entraînement ET peut être lancé directement
    run_script("chatbot_optimized.py")

    print("✅ Pipeline complet exécuté avec succès 🎉")
    print("✅ Tu peux maintenant tester le chatbot !\n")

if __name__ == "__main__":
    main()