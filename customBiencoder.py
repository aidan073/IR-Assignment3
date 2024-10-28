from sentence_transformers import SentenceTransformer
import torch

class CustomBiencoder():
    def __init__(self, model_name:str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        self.model = model
    def getModel(self):
        return self.model
    def setModel(self, model) -> None:
        self.model = model