import os
import math
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def compute_perplexity(model, tokenizer, texts, device):
    total_loss, total_tokens = 0.0, 0
    for example in texts:
        enc = tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        input_ids = enc.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        tokens = input_ids.numel()
        total_loss += loss * tokens
        total_tokens += tokens
    return math.exp(total_loss / total_tokens)

def main():
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Charge modèle et tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("output")
    model = GPT2LMHeadModel.from_pretrained("output").to(device)
    model.eval()

    # 3. Charge les données de validation comme test
    data_files = {"test": "data/valid.txt"}
    dataset = load_dataset("text", data_files=data_files)["test"]

    # 4. Calcule la perplexité
    ppl = compute_perplexity(model, tokenizer, dataset, device)
    print(f"Perplexité sur valid.txt : {ppl:.2f}")

if __name__ == "__main__":
    main()
