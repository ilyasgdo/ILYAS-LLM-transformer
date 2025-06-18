import os
import torch
import math
import json
from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def calculate_perplexity(model, dataloader, device):
    """
    Calcule la perplexité d'un modèle sur un dataset
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calcul de la perplexité"):
            # Déplacer les données sur le device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Calculer le nombre de tokens valides
            valid_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    # Calculer la perplexité moyenne
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss

def evaluate_model_detailed(model_path, test_file, batch_size=8):
    """
    Évaluation détaillée d'un modèle avec métriques avancées
    """
    print(f"🔍 Évaluation du modèle: {model_path}")
    print("=" * 60)
    
    # Vérification du device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device utilisé: {device}")
    
    # Chargement du modèle et tokenizer
    print("📂 Chargement du modèle...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device)
        print(f"✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None
    
    # Chargement des données de test
    print("📊 Chargement des données de test...")
    if not os.path.exists(test_file):
        print(f"❌ Fichier de test non trouvé: {test_file}")
        return None
    
    test_dataset = load_dataset('text', data_files={'test': test_file})['test']
    print(f"📈 {len(test_dataset)} exemples de test")
    
    # Tokenisation
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=1024,
            padding=True,
            return_tensors='pt'
        )
    
    tokenized_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Calcul de la perplexité
    print("🧮 Calcul de la perplexité...")
    perplexity, avg_loss = calculate_perplexity(model, dataloader, device)
    
    # Métriques du modèle
    model_size = sum(p.numel() for p in model.parameters())
    model_size_mb = model_size * 4 / (1024 * 1024)  # Approximation en MB
    
    # Résultats
    results = {
        'model_path': model_path,
        'perplexity': perplexity,
        'average_loss': avg_loss,
        'model_parameters': model_size,
        'model_size_mb': model_size_mb,
        'test_samples': len(test_dataset),
        'device': str(device)
    }
    
    # Affichage des résultats
    print("\n📊 RÉSULTATS D'ÉVALUATION")
    print("=" * 40)
    print(f"🎯 Perplexité: {perplexity:.2f}")
    print(f"📉 Loss moyenne: {avg_loss:.4f}")
    print(f"🧮 Paramètres: {model_size/1e6:.1f}M")
    print(f"💾 Taille modèle: {model_size_mb:.1f} MB")
    print(f"📈 Échantillons test: {len(test_dataset)}")
    
    # Interprétation de la perplexité
    print("\n🎭 INTERPRÉTATION")
    print("=" * 40)
    if perplexity < 20:
        print("🌟 Excellent! Perplexité très faible")
    elif perplexity < 50:
        print("✅ Bon modèle, perplexité acceptable")
    elif perplexity < 100:
        print("⚠️ Modèle moyen, peut être amélioré")
    else:
        print("❌ Perplexité élevée, modèle à retravailler")
    
    return results

def compare_models(model_paths, test_file):
    """
    Compare plusieurs modèles sur la même base de test
    """
    print("🏆 COMPARAISON DE MODÈLES")
    print("=" * 60)
    
    all_results = []
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            results = evaluate_model_detailed(model_path, test_file)
            if results:
                all_results.append(results)
            print("\n" + "-" * 60 + "\n")
        else:
            print(f"⚠️ Modèle non trouvé: {model_path}")
    
    if len(all_results) > 1:
        print("🏅 CLASSEMENT PAR PERPLEXITÉ")
        print("=" * 40)
        
        # Tri par perplexité croissante (meilleur = plus faible)
        sorted_results = sorted(all_results, key=lambda x: x['perplexity'])
        
        for i, result in enumerate(sorted_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{medal} {os.path.basename(result['model_path'])}: {result['perplexity']:.2f}")
        
        # Sauvegarde des résultats
        with open('model_comparison.json', 'w') as f:
            json.dump(sorted_results, f, indent=2)
        print("\n💾 Résultats sauvegardés dans: model_comparison.json")
    
    return all_results

def main():
    """
    Script principal d'évaluation
    """
    print("🎯 ÉVALUATEUR DE PERPLEXITÉ")
    print("=" * 60)
    
    # Fichier de test
    test_file = "data/test.txt"
    if not os.path.exists(test_file):
        # Utiliser le fichier de validation si pas de test
        test_file = "data/valid.txt"
        if not os.path.exists(test_file):
            print("❌ Aucun fichier de test trouvé (test.txt ou valid.txt)")
            return
        print(f"ℹ️ Utilisation du fichier de validation: {test_file}")
    
    # Modèles à évaluer
    model_paths = [
        "output",  # Modèle original
        "output_best_perplexity",  # Modèle optimisé
    ]
    
    # Ajouter les checkpoints récents si disponibles
    if os.path.exists("output"):
        checkpoints = [d for d in os.listdir("output") if d.startswith("checkpoint-")]
        if checkpoints:
            # Prendre le checkpoint le plus récent
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            model_paths.append(os.path.join("output", latest_checkpoint))
    
    # Filtrer les modèles existants
    existing_models = [path for path in model_paths if os.path.exists(path)]
    
    if not existing_models:
        print("❌ Aucun modèle trouvé à évaluer")
        print("💡 Assurez-vous d'avoir entraîné un modèle d'abord")
        return
    
    print(f"📋 Modèles à évaluer: {len(existing_models)}")
    for model in existing_models:
        print(f"  - {model}")
    
    # Évaluation
    if len(existing_models) == 1:
        evaluate_model_detailed(existing_models[0], test_file)
    else:
        compare_models(existing_models, test_file)

if __name__ == "__main__":
    main()