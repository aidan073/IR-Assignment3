import numpy as np
import torch
import csv
from sentence_transformers import SentenceTransformer, losses, evaluation
from torch.utils.data import DataLoader
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

    def loadModel(self, path) -> None:
        self.model = SentenceTransformer(path)

    def getEmbeddings(self, batch, outfile_name=None):
        embeddings = self.model.encode(batch, batch_size=32, show_progress_bar=True)
        if outfile_name:
            np.save(outfile_name, embeddings)
        return embeddings
    
    def loadEmbeddings(self, embedding_path):
        embeddings = np.load(embedding_path)
        return embeddings
    
    def fineTune(self, train_examples, validation_examples, num_epochs = 1):
        train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=32)
        train_loss = losses.CoSENTLoss(model=self.model)
        sentences1 = []
        sentences2 = []
        scores = []
        for example in validation_examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        dev_evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1=sentences1, sentences2=sentences2, scores=scores, main_similarity=evaluation.SimilarityFunction.COSINE, name="sts-dev")
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=100, evaluator=dev_evaluator, evaluation_steps=None)

    def writeTopN(self, queries, collection, run_name:str, output_path:str, top_n:int = 100, q_map:dict = None, d_map:dict = None):
        similarities = cosine_similarity(queries, collection)
        with open(output_path, "w", newline = '') as f:
            writer = csv.writer(f, delimiter='\t')
            for i in range(len(queries)):
                top_indices = np.argsort(similarities[i])[::-1][:top_n]
                for rank, j in enumerate(top_indices):
                    writer.writerow([q_map[i], 'Q0', d_map[j], rank+1, similarities[i][j], f"{run_name}"])
                