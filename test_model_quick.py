import os
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import json
import time

def quick_model_test(model_path="output_best_perplexity"):
    """Test rapide du modèle entraîné"""
    print("🧪 Test rapide du modèle")
    print("=" * 30)
    
    # Vérification de l'existence du modèle
    if not os.path.exists(model_path):
        print(f"❌ Erreur: Le modèle {model_path} n'existe pas")
        return
    
    print(f"📂 Chargement du modèle depuis: {model_path}")
    
    try:
        # Chargement
        start_time = time.time()
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        load_time = time.time() - start_time
        print(f"✅ Modèle chargé en {load_time:.2f}s sur {device}")
        
        # Informations du modèle
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"🧮 Paramètres: {num_params:.1f}M")
        
        # Affichage des métriques d'entraînement
        metrics_file = os.path.join(model_path, "training_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            print(f"📊 Perplexité d'entraînement: {metrics.get('final_perplexity', 'N/A')}")
            print(f"📉 Loss finale: {metrics.get('final_loss', 'N/A')}")
        
        # Tests de génération
        test_prompts = [
            "Bonjour, comment",
            "La technologie",
            "Il était une fois",
            "Dans le futur",
            "L'intelligence artificielle"
        ]
        
        print("\n🎯 Tests de génération:")
        print("-" * 40)
        
        total_gen_time = 0
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Prompt: '{prompt}'")
            
            # Génération
            start_gen = time.time()
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=50,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            gen_time = time.time() - start_gen
            total_gen_time += gen_time
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   Résultat: {generated_text}")
            print(f"   Temps: {gen_time:.2f}s")
        
        # Statistiques finales
        avg_gen_time = total_gen_time / len(test_prompts)
        print(f"\n📈 Statistiques:")
        print(f"⏱️ Temps moyen de génération: {avg_gen_time:.2f}s")
        print(f"🚀 Vitesse: {50/avg_gen_time:.1f} tokens/s")
        
        # Test de mémoire GPU
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 Mémoire GPU: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")
        
        print("\n✅ Test terminé avec succès !")
        
    except Exception as e:
        print(f"❌ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()

def compare_models():
    """Compare les différents modèles disponibles"""
    print("🔍 Comparaison des modèles")
    print("=" * 30)
    
    model_dirs = [
        "output_best_perplexity",
        "output_optimized",
        "v1output"
    ]
    
    results = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"\n📂 Analyse de {model_dir}:")
            
            # Lecture des métriques
            metrics_files = [
                os.path.join(model_dir, "training_metrics.json"),
                os.path.join(model_dir, "final_metrics.json")
            ]
            
            metrics = None
            for metrics_file in metrics_files:
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    break
            
            if metrics:
                perplexity = metrics.get('final_perplexity', 'N/A')
                loss = metrics.get('final_loss', 'N/A')
                params = metrics.get('model_parameters', 'N/A')
                
                print(f"  📊 Perplexité: {perplexity}")
                print(f"  📉 Loss: {loss}")
                print(f"  🧮 Paramètres: {params}M")
                
                results.append({
                    'model': model_dir,
                    'perplexity': perplexity,
                    'loss': loss,
                    'parameters': params
                })
            else:
                print(f"  ⚠️ Pas de métriques trouvées")
        else:
            print(f"  ❌ {model_dir} n'existe pas")
    
    # Sauvegarde de la comparaison
    if results:
        with open("model_comparison.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Comparaison sauvegardée dans model_comparison.json")
        
        # Affichage du meilleur modèle
        valid_results = [r for r in results if isinstance(r['perplexity'], (int, float))]
        if valid_results:
            best_model = min(valid_results, key=lambda x: x['perplexity'])
            print(f"\n🏆 Meilleur modèle: {best_model['model']}")
            print(f"   Perplexité: {best_model['perplexity']}")

def main():
    """Menu principal"""
    print("🧪 Test et comparaison des modèles")
    print("=" * 40)
    
    while True:
        print("\n📋 Options:")
        print("1. Test rapide du modèle principal")
        print("2. Comparaison des modèles")
        print("3. Test d'un modèle spécifique")
        print("4. Quitter")
        
        choice = input("\n🎯 Votre choix (1-4): ").strip()
        
        if choice == '1':
            quick_model_test()
        elif choice == '2':
            compare_models()
        elif choice == '3':
            model_path = input("📂 Chemin du modèle: ").strip()
            if model_path:
                quick_model_test(model_path)
            else:
                print("❌ Chemin invalide")
        elif choice == '4':
            print("👋 Au revoir !")
            break
        else:
            print("❌ Choix invalide")

if __name__ == "__main__":
    main()