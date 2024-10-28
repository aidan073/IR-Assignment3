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
    def writeTopN(queries, collection, top_n, q_map, d_map, output_file):
        similarities = cosine_similarity(queries, collection)
        query_results = []
        with open(output_file, "w", newline = '') as f:
            writer = csv.writer(f)
            for i in len(queries):
                top_indices = np.argsort(similarities[i])[::-1][:top_n]
                query_results.append([(q_map[i], similarities[i][j]) for j in top_indices])