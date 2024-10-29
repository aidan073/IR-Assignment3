import numpy as np
import torch
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class CustomBiencoder():
    def __init__(self, model_name:str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        self.model = model
    def getModel(self):
        return self.model
    def setModel(self, model) -> None:
        self.model = model
    def getEmbeddings(self, batch):
        embeddings = self.model.encode(batch, batch_size=32, show_progress_bar=True)
        return embeddings
    def writeTopN(self, queries, collection, q_map:dict, d_map:dict, run_name:str, output_path:str, top_n:int = 100):
        similarities = cosine_similarity(queries, collection)
        with open(output_path, "w", newline = '') as f:
            writer = csv.writer(f, delimiter='\t')
            for i in range(len(queries)):
                top_indices = np.argsort(similarities[i])[::-1][:top_n]
                for rank, j in enumerate(top_indices):
                    writer.writerow([q_map[i], 'Q0', d_map[j], rank+1, similarities[i][j], f"{run_name}"])
                