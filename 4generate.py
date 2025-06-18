import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def main():
    # 1. Spécifie le device (GPU si dispo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Charge le tokenizer et le modèle depuis le dossier output/
    tokenizer = GPT2TokenizerFast.from_pretrained("output")
    model = GPT2LMHeadModel.from_pretrained("output").to(device)
    model.eval()

    # 3. Phrase de départ
    prompt = "Aujourd'hui, je vais te raconter une histoire :"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 4. Génération
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=100,          # longueur totale (prompt + continuation)
            do_sample=True,          # sampling
            top_k=50,                # génère à partir des 50 tokens les plus probables
            top_p=0.95,              # nucleus sampling
            temperature=0.8,         # contrôle de créativité
            num_return_sequences=1   # nombre de textes à générer
        )

    # 5. Décoder et afficher
    for i, output_ids in enumerate(outputs):
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"\n=== Génération #{i+1} ===\n{text}\n")

if __name__ == "__main__":
    main()
