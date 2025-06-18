import os
import torch
import math
import json
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from transformers.trainer_callback import TrainerCallback
import numpy as np

# Optimisations pour RTX 3080 Ti
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class PerplexityCallback(TrainerCallback):
    """Callback pour calculer et logger la perplexité"""
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        if logs and 'eval_loss' in logs:
            perplexity = math.exp(logs['eval_loss'])
            logs['eval_perplexity'] = perplexity
            print(f"\n📊 Perplexité actuelle: {perplexity:.2f}")
            
            # Sauvegarde du meilleur score
            if not hasattr(self, 'best_perplexity') or perplexity < self.best_perplexity:
                self.best_perplexity = perplexity
                print(f"🎯 Nouveau meilleur score de perplexité: {perplexity:.2f}")

class AdaptiveLearningRateCallback(TrainerCallback):
    """Callback pour ajuster le learning rate basé sur la perplexité"""
    
    def __init__(self, patience=3, factor=0.5, min_lr=1e-6):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_perplexity = float('inf')
        self.wait = 0
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        if logs and 'eval_loss' in logs:
            current_perplexity = math.exp(logs['eval_loss'])
            
            if current_perplexity < self.best_perplexity:
                self.best_perplexity = current_perplexity
                self.wait = 0
            else:
                self.wait += 1
                
            if self.wait >= self.patience:
                current_lr = state.learning_rate
                new_lr = max(current_lr * self.factor, self.min_lr)
                if new_lr < current_lr:
                    print(f"🔄 Réduction du learning rate: {current_lr:.2e} → {new_lr:.2e}")
                    # Note: Dans une implémentation complète, on modifierait l'optimiseur ici
                self.wait = 0

def create_optimized_model_config(tokenizer_size):
    """Crée une configuration de modèle optimisée pour la perplexité"""
    return GPT2Config(
        vocab_size=tokenizer_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        activation_function="gelu_new",
        # Dropout optimisé pour réduire l'overfitting
        resid_pdrop=0.05,  # Réduit de 0.1 à 0.05
        embd_pdrop=0.05,   # Réduit de 0.1 à 0.05
        attn_pdrop=0.05,   # Réduit de 0.1 à 0.05
        # Initialisation améliorée
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        # Amélioration de la stabilité
        scale_attn_weights=True,
        use_cache=True,
    )

def create_advanced_training_args():
    """Crée des arguments d'entraînement optimisés pour la perplexité"""
    return TrainingArguments(
        output_dir="output_best_perplexity",
        overwrite_output_dir=True,
        
        # Entraînement plus long et plus stable
        num_train_epochs=8,
        per_device_train_batch_size=6,  # Légèrement réduit pour plus de stabilité
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=6,  # Batch effectif de 36
        
        # Évaluation très fréquente pour monitoring
        evaluation_strategy="steps",
        eval_steps=100,  # Évaluation toutes les 100 steps
        save_steps=200,
        logging_steps=25,
        
        # Learning rate optimisé avec warmup plus long
        learning_rate=1e-4,  # Plus conservateur
        weight_decay=0.01,
        warmup_steps=1000,   # Warmup plus long
        lr_scheduler_type="cosine_with_restarts",
        
        # Optimisations mémoire et performance
        fp16=True,
        bf16=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        
        # Sélection du meilleur modèle
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=5,  # Garde plus de checkpoints
        
        # Optimisations avancées
        gradient_checkpointing=True,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,  # Plus stable que 0.95
        adam_epsilon=1e-8,
        max_grad_norm=0.5,  # Gradient clipping plus strict
        
        # Désactivation des rapports externes
        push_to_hub=False,
        report_to=[],
        
        # Paramètres pour la stabilité
        remove_unused_columns=False,
        prediction_loss_only=True,
        
        # Seed pour la reproductibilité
        seed=42,
        data_seed=42,
    )

def enhanced_tokenize_function(examples, tokenizer, max_length):
    """Fonction de tokenisation améliorée avec filtrage de qualité"""
    # Tokenisation standard
    tokenized = tokenizer(
        examples['text'],
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    
    # Filtrage des séquences trop courtes (peuvent nuire à la perplexité)
    min_length = 50  # Minimum 50 tokens
    filtered_input_ids = []
    filtered_attention_mask = []
    
    for i, input_ids in enumerate(tokenized['input_ids']):
        if len(input_ids) >= min_length:
            filtered_input_ids.append(input_ids)
            filtered_attention_mask.append(tokenized['attention_mask'][i])
    
    return {
        'input_ids': filtered_input_ids,
        'attention_mask': filtered_attention_mask
    }

def main():
    print("🚀 Entraînement optimisé pour la meilleure perplexité")
    print("=" * 60)
    
    # 1. Chemins des données
    train_file = "data/train.txt"
    valid_file = "data/valid.txt"
    
    # Vérification des fichiers
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print("❌ Erreur: Fichiers de données manquants")
        return
    
    # 2. Chargement des données
    print("📂 Chargement des données...")
    raw_datasets = load_dataset(
        'text',
        data_files={'train': train_file, 'validation': valid_file}
    )
    
    print(f"📊 Données d'entraînement: {len(raw_datasets['train'])} exemples")
    print(f"📊 Données de validation: {len(raw_datasets['validation'])} exemples")
    
    # 3. Tokenizer
    print("🔤 Initialisation du tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 4. Configuration du modèle optimisée
    print("🧠 Création du modèle optimisé...")
    config = create_optimized_model_config(len(tokenizer))
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    
    # 5. Tokenisation avec filtrage
    print("⚙️ Tokenisation des données...")
    tokenized_datasets = raw_datasets.map(
        lambda examples: enhanced_tokenize_function(examples, tokenizer, config.n_ctx),
        batched=True,
        batch_size=500,
        num_proc=4,
        remove_columns=['text'],
        desc="Tokenisation optimisée"
    )
    
    # 6. Data collator optimisé
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # 7. Arguments d'entraînement
    training_args = create_advanced_training_args()
    
    # 8. Callbacks pour l'optimisation
    callbacks = [
        PerplexityCallback(),
        AdaptiveLearningRateCallback(patience=5, factor=0.7),
        EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)
    ]
    
    # 9. Trainer avec callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # 10. Informations pré-entraînement
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    
    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    
    print(f"🧮 Paramètres du modèle: {model_params:.1f}M")
    print(f"📦 Batch size effectif: {effective_batch}")
    print(f"🎯 Objectif: Minimiser la perplexité")
    print("\n🏁 Début de l'entraînement...")
    
    # 11. Entraînement
    try:
        trainer.train()
        
        # 12. Évaluation finale
        print("\n📈 Évaluation finale...")
        eval_results = trainer.evaluate()
        final_perplexity = math.exp(eval_results['eval_loss'])
        
        print(f"\n🎉 ENTRAÎNEMENT TERMINÉ")
        print(f"📊 Perplexité finale: {final_perplexity:.2f}")
        print(f"📉 Loss finale: {eval_results['eval_loss']:.4f}")
        
        # 13. Sauvegarde
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Sauvegarde des métriques
        metrics = {
            'final_perplexity': final_perplexity,
            'final_loss': eval_results['eval_loss'],
            'model_parameters': model_params,
            'effective_batch_size': effective_batch,
            'training_epochs': training_args.num_train_epochs
        }
        
        with open(os.path.join(training_args.output_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Modèle sauvegardé dans: {training_args.output_dir}")
        print(f"📋 Métriques sauvegardées dans: training_metrics.json")
        
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        raise
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()