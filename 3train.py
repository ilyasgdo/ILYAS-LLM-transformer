import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW

# Optimisations pour RTX 3080 Ti
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 1. Paths to your data files
train_file = "data/train.txt"
valid_file = "data/valid.txt"

# 2. Load raw text dataset
raw_datasets = load_dataset(
    'text',
    data_files={'train': train_file, 'validation': valid_file}
)

# 3. Initialize tokenizer
pretrained_tokenizer = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_tokenizer)
# Add a pad token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 4. Model configuration optimisée pour RTX 3080 Ti (12GB VRAM)
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=1024,  # Augmenté pour de meilleures performances
    n_ctx=1024,
    n_embd=768,        # Augmenté de 384 à 768
    n_layer=12,        # Augmenté de 6 à 12 couches
    n_head=12,         # Augmenté de 6 à 12 têtes d'attention
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)
# Instantiate model from scratch
model = GPT2LMHeadModel(config)
# Resize embeddings in case we've added new tokens (pad_token)
model.resize_token_embeddings(len(tokenizer))

# 5. Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        return_special_tokens_mask=True,
        truncation=True,
        max_length=config.n_ctx,
        padding=False,  # Pas de padding ici, fait par le data collator
    )

# 6. Apply tokenization avec optimisations
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    batch_size=1000,           # Batch size plus grand pour la tokenisation
    num_proc=4,                # Parallélisation
    remove_columns=['text'],
    desc="Tokenizing datasets"
)

# 7. Data collator optimisé pour causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,      # Optimisation pour les tensors cores
    return_tensors="pt"
)

# 8. Training arguments optimisés pour RTX 3080 Ti
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=5,                    # Augmenté pour un meilleur entraînement
    per_device_train_batch_size=8,         # Augmenté de 4 à 8
    per_device_eval_batch_size=8,          # Augmenté de 4 à 8
    gradient_accumulation_steps=4,         # Batch effectif de 32
    evaluation_strategy="steps",
    eval_steps=250,                        # Évaluation plus fréquente
    save_steps=250,                        # Sauvegarde plus fréquente
    logging_steps=50,                      # Logging plus fréquent
    learning_rate=3e-4,                    # Learning rate optimisé
    weight_decay=0.01,
    warmup_steps=500,                      # Plus de warmup steps
    lr_scheduler_type="cosine",            # Scheduler cosine
    fp16=True,                             # Precision mixte
    bf16=False,                            # BF16 désactivé pour RTX 3080 Ti
    dataloader_pin_memory=True,            # Optimisation mémoire
    dataloader_num_workers=4,              # Parallélisation du chargement
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,                    # Limite les checkpoints sauvegardés
    push_to_hub=False,
    report_to=[],                          # Désactive wandb/tensorboard par défaut
    # Optimisations mémoire spécifiques
    gradient_checkpointing=True,           # Trade compute for memory
    optim="adamw_torch",                   # Optimiseur optimisé
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
)

# 9. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 10. Train avec optimisations
if __name__ == "__main__":
    # Vérification de la disponibilité CUDA
    if torch.cuda.is_available():
        print(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire CUDA: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("ATTENTION: CUDA non disponible, entraînement sur CPU")
    
    # Nettoyage de la mémoire CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Taille du modèle: {sum(p.numel() for p in model.parameters())/1e6:.1f}M paramètres")
    print(f"Batch size effectif: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    # Entraînement
    trainer.train()
    
    # Sauvegarde du meilleur modèle
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("\n=== ENTRAÎNEMENT TERMINÉ ===")
    print(f"Modèle et tokenizer sauvegardés dans: {training_args.output_dir}")
    print(f"Nombre total d'époques: {training_args.num_train_epochs}")
    print(f"Paramètres du modèle: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Nettoyage final
    if torch.cuda.is_available():
        torch.cuda.empty_cache()