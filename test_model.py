#!/usr/bin/env python3
"""
Script de test pour le modèle LLM entraîné
Permet de tester la génération de texte et d'évaluer la qualité
"""

import os
import torch
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import time

def load_model(model_path="output_optimized"):
    """Charge le modèle et le tokenizer"""
    try:
        print(f"🔄 Chargement du modèle depuis {model_path}...")
        
        # Vérification de l'existence du modèle
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé dans {model_path}")
        
        # Chargement
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Configuration GPU si disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"✅ Modèle chargé sur {device}")
        print(f"🧠 Paramètres: {model.num_parameters():,}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, None, None

def generate_text(model, tokenizer, device, prompt, max_length=200, temperature=0.8, top_p=0.9):
    """Génère du texte à partir d'un prompt"""
    try:
        # Tokenisation du prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Génération
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        generation_time = time.time() - start_time
        
        # Décodage
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Statistiques
        tokens_generated = len(outputs[0]) - len(inputs[0])
        tokens_per_second = tokens_generated / generation_time
        
        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        return None

def interactive_test(model, tokenizer, device):
    """Mode interactif pour tester le modèle"""
    print("\n🎯 Mode interactif - Tapez 'quit' pour quitter")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n📝 Entrez votre prompt: ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt.strip():
                continue
            
            print("\n🔄 Génération en cours...")
            result = generate_text(model, tokenizer, device, prompt)
            
            if result:
                print(f"\n📖 Texte généré:")
                print("-" * 40)
                print(result['text'])
                print("-" * 40)
                print(f"⚡ {result['tokens_generated']} tokens en {result['generation_time']:.2f}s")
                print(f"🚀 Vitesse: {result['tokens_per_second']:.1f} tokens/sec")
            
        except KeyboardInterrupt:
            print("\n👋 Arrêt demandé")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

def benchmark_test(model, tokenizer, device):
    """Test de performance avec prompts prédéfinis"""
    print("\n🏁 Test de performance")
    print("=" * 30)
    
    test_prompts = [
        "Il était une fois",
        "La science moderne",
        "Dans un monde futuriste",
        "L'intelligence artificielle",
        "Au cœur de la forêt"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Test {i}/5: '{prompt}'")
        
        result = generate_text(model, tokenizer, device, prompt, max_length=150)
        
        if result:
            results.append(result)
            print(f"✅ {result['tokens_generated']} tokens - {result['tokens_per_second']:.1f} tok/s")
        else:
            print("❌ Échec")
    
    # Statistiques globales
    if results:
        avg_speed = sum(r['tokens_per_second'] for r in results) / len(results)
        total_tokens = sum(r['tokens_generated'] for r in results)
        total_time = sum(r['generation_time'] for r in results)
        
        print(f"\n📊 Résultats globaux:")
        print(f"   Vitesse moyenne: {avg_speed:.1f} tokens/sec")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Temps total: {total_time:.2f}s")
        
        return {
            "average_speed": avg_speed,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "individual_results": results
        }
    
    return None

def load_metrics(model_path="output_optimized"):
    """Charge les métriques d'entraînement si disponibles"""
    metrics_file = os.path.join(model_path, "final_metrics.json")
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            print("\n📈 Métriques d'entraînement:")
            print(f"   Perplexité finale: {metrics.get('final_perplexity', 'N/A'):.2f}")
            print(f"   Steps total: {metrics.get('total_steps', 'N/A'):,}")
            print(f"   Paramètres: {metrics.get('model_parameters', 'N/A'):,}")
            print(f"   GPU utilisé: {metrics.get('gpu_used', 'N/A')}")
            
            return metrics
        except Exception as e:
            print(f"⚠️ Impossible de charger les métriques: {e}")
    
    return None

def main():
    print("🧪 Test du modèle LLM entraîné")
    print("=" * 40)
    
    # Chargement du modèle
    model, tokenizer, device = load_model()
    
    if model is None:
        print("❌ Impossible de charger le modèle")
        return
    
    # Chargement des métriques
    load_metrics()
    
    # Menu principal
    while True:
        print("\n🎯 Options disponibles:")
        print("1. Test interactif")
        print("2. Benchmark de performance")
        print("3. Test rapide")
        print("4. Quitter")
        
        choice = input("\nChoisissez une option (1-4): ").strip()
        
        if choice == '1':
            interactive_test(model, tokenizer, device)
        
        elif choice == '2':
            benchmark_results = benchmark_test(model, tokenizer, device)
            if benchmark_results:
                # Sauvegarde des résultats
                with open("benchmark_results.json", "w") as f:
                    json.dump(benchmark_results, f, indent=2)
                print("💾 Résultats sauvegardés dans benchmark_results.json")
        
        elif choice == '3':
            print("\n🚀 Test rapide...")
            result = generate_text(model, tokenizer, device, "Bonjour, je suis")
            if result:
                print(f"\n📖 Résultat: {result['text']}")
                print(f"⚡ {result['tokens_per_second']:.1f} tokens/sec")
        
        elif choice == '4':
            print("👋 Au revoir!")
            break
        
        else:
            print("❌ Option invalide")

if __name__ == "__main__":
    main()