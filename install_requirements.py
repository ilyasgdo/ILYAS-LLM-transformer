#!/usr/bin/env python3
"""
Script d'installation automatique des dépendances pour l'entraînement LLM optimisé
Optimisé pour RTX 3080 Ti avec 12GB VRAM
"""

import subprocess
import sys
import os

def install_package(package):
    """Installe un package avec pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installé avec succès")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Erreur lors de l'installation de {package}")
        return False

def check_cuda():
    """Vérifie la disponibilité de CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🎮 CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️ CUDA non disponible - utilisation du CPU")
            return False
    except ImportError:
        print("❌ PyTorch non installé")
        return False

def main():
    print("🚀 Installation des dépendances pour l'entraînement LLM optimisé")
    print("=" * 60)
    
    # Packages requis avec versions optimisées
    packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "datasets>=2.12.0",
        "tokenizers>=0.13.0",
        "accelerate>=0.20.0",
        "tensorboard",
        "numpy",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ]
    
    print("📦 Installation des packages...")
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    
    if failed_packages:
        print(f"❌ Échec de l'installation: {', '.join(failed_packages)}")
        print("Essayez d'installer manuellement avec:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
    else:
        print("✅ Toutes les dépendances installées avec succès!")
    
    # Vérification CUDA
    print("\n🔍 Vérification de l'environnement...")
    check_cuda()
    
    # Création des dossiers nécessaires
    dirs_to_create = ["data", "output_optimized", "logs", "cache"]
    for dir_name in dirs_to_create:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Dossier créé: {dir_name}")
    
    print("\n🎯 Installation terminée!")
    print("Vous pouvez maintenant lancer l'entraînement avec: python 3train.py")
    print("Pour surveiller l'entraînement: tensorboard --logdir=./logs")

if __name__ == "__main__":
    main()