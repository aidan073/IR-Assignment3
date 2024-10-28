from sentence_transformers import CrossEncoder, util
import torch

class CustomCrossencoder():
    def __init__(self, model_name:str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(model_name, device=device)
        self.model = model
    def getModel(self):
        return self.model
    def setModel(self, model) -> None:
        self.model = model
    def cosineSim(self, emb1, emb2) -> torch.Tensor:
        return util.cos_sim(emb1, emb2)
