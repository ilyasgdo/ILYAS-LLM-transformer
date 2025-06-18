import os
import torch
import math
import json
import gc
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
from torch.utils.data import DataLoader
import time

# Optimisations GPU pour utilisation stable
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configuration pour éviter les pics GPU
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)  # Limite à 95% de la VRAM
    torch.cuda.memory.set_per_process_memory_fraction(0.95)

class StableGPUCallback(TrainerCallback):
    """Callback pour maintenir une utilisation GPU stable"""
    
    def __init__(self, memory_cleanup_steps=50):
        self.memory_cleanup_steps = memory_cleanup_steps
        self.step_count = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_count += 1
        
        # Nettoyage mémoire périodique
        if self.step_count % self.memory_cleanup_steps == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        # Nettoyage après évaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

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

class GPUMonitorCallback(TrainerCallback):
    """Callback pour monitorer l'utilisation GPU"""
    
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0 and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            utilization = (memory_allocated / memory_total) * 100
            
            print(f"🎮 GPU Step {self.step_count}: {utilization:.1f}% ({memory_allocated:.1f}GB/{memory_total:.1f}GB)")

def create_stable_model_config(tokenizer_size):
    """Configuration modèle optimisée pour stabilité GPU"""
    return GPT2Config(
        vocab_size=tokenizer_size,
        n_positions=512,  # Réduit pour stabilité
        n_ctx=512,        # Réduit pour stabilité
        n_embd=512,       # Réduit pour éviter les pics
        n_layer=8,        # Réduit pour stabilité
        n_head=8,         # Réduit pour stabilité
        activation_function="gelu_new",
        # Dropout optimisé
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        # Initialisation stable
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        use_cache=False,  # Désactivé pour économiser la mémoire
    )

def create_stable_training_args():
    """Arguments d'entraînement pour utilisation GPU stable"""
    return TrainingArguments(
        output_dir="output_stable_gpu",
        overwrite_output_dir=True,
        
        # Configuration batch pour stabilité
        num_train_epochs=5,
        per_device_train_batch_size=4,  # Plus petit pour stabilité
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Compense la réduction du batch
        
        # Évaluation et sauvegarde
        evaluation_strategy="steps",
        eval_steps=200,  # Moins fréquent pour réduire les pics
        save_steps=400,
        logging_steps=50,
        
        # Learning rate conservateur
        learning_rate=5e-5,  # Plus conservateur
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="linear",  # Plus stable que cosine
        
        # Optimisations mémoire strictes
        fp16=True,
        bf16=False,
        dataloader_pin_memory=False,  # Désactivé pour éviter les pics
        dataloader_num_workers=2,     # Réduit pour stabilité
        
        # Gestion modèle
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,  # Limite les sauvegardes
        
        # Optimisations GPU
        gradient_checkpointing=True,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        
        # Stabilité
        push_to_hub=False,
        report_to=[],
        remove_unused_columns=True,
        prediction_loss_only=True,
        
        # Reproductibilité
        seed=42,
        data_seed=42,
        
        # Paramètres pour éviter les pics
        dataloader_drop_last=True,  # Évite les batchs de taille variable
        ignore_data_skip=True,
    )

def preload_and_cache_data(tokenized_datasets, batch_size=4):
    """Précharge les données pour éviter les pics de chargement"""
    print("🔄 Préchargement des données pour stabiliser l'utilisation GPU...")
    
    # Précharge quelques batchs en mémoire
    train_loader = DataLoader(
        tokenized_datasets['train'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Précharge les premiers batchs
    preloaded_batches = []
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Précharge 10 batchs
            break
        preloaded_batches.append(batch)
    
    print(f"✅ {len(preloaded_batches)} batchs préchargés")
    return preloaded_batches

def enhanced_tokenize_function(examples, tokenizer, max_length):
    """Tokenisation optimisée pour stabilité"""
    tokenized = tokenizer(
        examples['text'],
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",  # Padding fixe pour éviter les variations
        return_tensors=None
    )
    
    return tokenized

def main():
    print("🚀 Entraînement avec utilisation GPU stable")
    print("=" * 60)
    
    # 1. Vérification GPU
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible")
        return
    
    device = torch.device("cuda")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Nettoyage initial
    torch.cuda.empty_cache()
    gc.collect()
    
    # 2. Chemins des données
    train_file = "data/train.txt"
    valid_file = "data/valid.txt"
    
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print("❌ Erreur: Fichiers de données manquants")
        return
    
    # 3. Chargement des données
    print("📂 Chargement des données...")
    raw_datasets = load_dataset(
        'text',
        data_files={'train': train_file, 'validation': valid_file},
        cache_dir="./cache"
    )
    
    print(f"📊 Données d'entraînement: {len(raw_datasets['train'])} exemples")
    print(f"📊 Données de validation: {len(raw_datasets['validation'])} exemples")
    
    # 4. Tokenizer
    print("🔤 Initialisation du tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 5. Configuration modèle stable
    print("🧠 Création du modèle stable...")
    config = create_stable_model_config(len(tokenizer))
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    # 6. Tokenisation
    print("⚙️ Tokenisation des données...")
    tokenized_datasets = raw_datasets.map(
        lambda examples: enhanced_tokenize_function(examples, tokenizer, config.n_ctx),
        batched=True,
        batch_size=100,  # Plus petit pour stabilité
        num_proc=2,      # Réduit pour éviter les pics
        remove_columns=['text'],
        desc="Tokenisation stable"
    )
    
    # 7. Préchargement des données
    preloaded_batches = preload_and_cache_data(tokenized_datasets, batch_size=4)
    
    # 8. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # 9. Arguments d'entraînement
    training_args = create_stable_training_args()
    
    # 10. Callbacks pour stabilité
    callbacks = [
        StableGPUCallback(memory_cleanup_steps=50),
        PerplexityCallback(),
        GPUMonitorCallback(log_interval=100),
        EarlyStoppingCallback(early_stopping_patience=5)
    ]
    
    # 11. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # 12. Informations pré-entraînement
    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    
    print(f"🧮 Paramètres du modèle: {model_params:.1f}M")
    print(f"📦 Batch size effectif: {effective_batch}")
    print(f"🎯 Objectif: Utilisation GPU stable")
    
    # Vérification mémoire initiale
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        print(f"💾 Mémoire GPU utilisée: {memory_allocated:.1f} GB")
    
    print("\n🏁 Début de l'entraînement stable...")
    
    # 13. Entraînement
    try:
        trainer.train()
        
        # 14. Évaluation finale
        print("\n📈 Évaluation finale...")
        eval_results = trainer.evaluate()
        final_perplexity = math.exp(eval_results['eval_loss'])
        
        print(f"\n🎉 ENTRAÎNEMENT TERMINÉ")
        print(f"📊 Perplexité finale: {final_perplexity:.2f}")
        print(f"📉 Loss finale: {eval_results['eval_loss']:.4f}")
        
        # 15. Sauvegarde
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Métriques finales
        metrics = {
            'final_perplexity': final_perplexity,
            'final_loss': eval_results['eval_loss'],
            'model_parameters': model_params,
            'effective_batch_size': effective_batch,
            'max_memory_used': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        }
        
        with open(os.path.join(training_args.output_dir, 'stable_training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Modèle sauvegardé dans: {training_args.output_dir}")
        
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        raise
    
    finally:
        # Nettoyage final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Nettoyage mémoire effectué")

if __name__ == "__main__":
    main()