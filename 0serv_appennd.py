from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import os

# Remplace ce chemin par ton chemin réel
FICHIER_CIBLE = r"C:\Users\ilyas\Documents\PROJET LLM\data\all.txt"
SEUIL_MAX = 40000

app = FastAPI()

class PhraseRequest(BaseModel):
    text: str

def compter_lignes():
    if not os.path.exists(FICHIER_CIBLE):
        return 0
    with open(FICHIER_CIBLE, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

@app.post("/append")
async def append_line(request: PhraseRequest):
    try:
        nb_lignes = compter_lignes()
        if nb_lignes >= SEUIL_MAX:
            return {"status": "finished", "message": f"Limite de {SEUIL_MAX} phrases atteinte."}
        
        with open(FICHIER_CIBLE, 'a', encoding='utf-8') as f:
            f.write(request.text.strip() + '\n')
        return {"status": "success", "message": "Phrase ajoutée.", "current_count": nb_lignes + 1}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
