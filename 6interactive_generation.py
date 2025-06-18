import os
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import json

def load_trained_model(model_path="output_best_perplexity"):
    """Charge le modèle entraîné"""
    print(f"🔄 Chargement du modèle depuis {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"❌ Erreur: Le dossier {model_path} n'existe pas")
        return None, None
    
    try:
        # Chargement du tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        
        # Chargement du modèle
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Configuration pour la génération
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        print(f"✅ Modèle chargé avec succès sur {device}")
        
        # Affichage des métriques si disponibles
        metrics_file = os.path.join(model_path, "training_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            print(f"📊 Perplexité du modèle: {metrics.get('final_perplexity', 'N/A')}")
            print(f"🧮 Paramètres: {metrics.get('model_parameters', 'N/A')}M")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, None

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.9, num_return_sequences=1):
    """Génère du texte à partir d'un prompt"""
    device = model.device
    
    # Encodage du prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Génération
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2
        )
    
    # Décodage des résultats
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

def interactive_generation():
    """Interface interactive pour la génération de texte"""
    print("🤖 Générateur de texte interactif")
    print("=" * 50)
    
    # Chargement du modèle
    model, tokenizer = load_trained_model()
    if model is None or tokenizer is None:
        return
    
    print("\n📝 Instructions:")
    print("- Tapez votre prompt et appuyez sur Entrée")
    print("- Tapez 'quit' pour quitter")
    print("- Tapez 'config' pour modifier les paramètres")
    print("- Tapez 'help' pour l'aide")
    
    # Paramètres par défaut
    config = {
        'max_length': 150,
        'temperature': 0.8,
        'top_p': 0.9,
        'num_sequences': 1
    }
    
    while True:
        print("\n" + "-" * 50)
        prompt = input("🎯 Votre prompt: ").strip()
        
        if prompt.lower() == 'quit':
            print("👋 Au revoir !")
            break
        
        elif prompt.lower() == 'help':
            print("\n📚 Aide:")
            print("- Écrivez un début de phrase ou de paragraphe")
            print("- Le modèle continuera votre texte")
            print("- Exemples de prompts:")
            print("  * 'Il était une fois'")
            print("  * 'La technologie moderne'")
            print("  * 'Dans un monde futuriste'")
            continue
        
        elif prompt.lower() == 'config':
            print("\n⚙️ Configuration actuelle:")
            print(f"- Longueur max: {config['max_length']}")
            print(f"- Température: {config['temperature']}")
            print(f"- Top-p: {config['top_p']}")
            print(f"- Nombre de séquences: {config['num_sequences']}")
            
            try:
                new_length = input(f"Nouvelle longueur max ({config['max_length']}): ").strip()
                if new_length:
                    config['max_length'] = int(new_length)
                
                new_temp = input(f"Nouvelle température ({config['temperature']}): ").strip()
                if new_temp:
                    config['temperature'] = float(new_temp)
                
                new_top_p = input(f"Nouveau top-p ({config['top_p']}): ").strip()
                if new_top_p:
                    config['top_p'] = float(new_top_p)
                
                new_num = input(f"Nombre de séquences ({config['num_sequences']}): ").strip()
                if new_num:
                    config['num_sequences'] = int(new_num)
                
                print("✅ Configuration mise à jour")
            except ValueError:
                print("❌ Valeur invalide, configuration inchangée")
            continue
        
        elif not prompt:
            print("⚠️ Veuillez entrer un prompt")
            continue
        
        # Génération
        print("\n🔄 Génération en cours...")
        try:
            generated_texts = generate_text(
                model, tokenizer, prompt,
                max_length=config['max_length'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                num_return_sequences=config['num_sequences']
            )
            
            print("\n📝 Texte(s) généré(s):")
            for i, text in enumerate(generated_texts, 1):
                print(f"\n--- Séquence {i} ---")
                print(text)
                print(f"--- Fin séquence {i} ---")
        
        except Exception as e:
            print(f"❌ Erreur lors de la génération: {e}")

def batch_generation():
    """Génération en lot à partir d'une liste de prompts"""
    print("📦 Génération en lot")
    print("=" * 30)
    
    # Chargement du modèle
    model, tokenizer = load_trained_model()
    if model is None or tokenizer is None:
        return
    
    # Prompts prédéfinis
    prompts = [
        "Il était une fois",
        "La science moderne",
        "Dans un futur proche",
        "L'intelligence artificielle",
        "Au cœur de la forêt"
    ]
    
    print("\n🎯 Génération pour les prompts prédéfinis:")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}: '{prompt}' ---")
        try:
            generated = generate_text(model, tokenizer, prompt, max_length=100)
            print(generated[0])
            results.append({
                'prompt': prompt,
                'generated': generated[0]
            })
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    # Sauvegarde des résultats
    output_file = "generated_texts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Résultats sauvegardés dans {output_file}")

def main():
    """Menu principal"""
    print("🚀 Utilisation du modèle entraîné")
    print("=" * 40)
    
    while True:
        print("\n📋 Options disponibles:")
        print("1. Génération interactive")
        print("2. Génération en lot")
        print("3. Quitter")
        
        choice = input("\n🎯 Votre choix (1-3): ").strip()
        
        if choice == '1':
            interactive_generation()
        elif choice == '2':
            batch_generation()
        elif choice == '3':
            print("👋 Au revoir !")
            break
        else:
            print("❌ Choix invalide, veuillez choisir 1, 2 ou 3")

if __name__ == "__main__":
    main()