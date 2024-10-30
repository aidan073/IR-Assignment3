import numpy as np
import torch
import csv
from sentence_transformers import SentenceTransformer, losses, evaluation
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

class BiEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, queries, responses, labels):
        self.queries = queries
        self.responses = responses
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        response = self.responses[idx]
        label = self.labels[idx]
        return query, response, label

class CustomBiencoder():
    def __init__(self, model_name:str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        self.model = model

    def getModel(self):
        return self.model
    
    def setModel(self, model) -> None:
        self.model = model

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
        val_dataloader = DataLoader(validation_examples, shuffle=False, batch_size=32)
        train_loss = losses.CosineSimilarityLoss(model=self.model)

        for epoch in range(num_epochs):

            # train
            self.model.train()
            total_train_loss = 0
            
            for batch in train_dataloader:
                self.model.zero_grad()
                loss_value = train_loss(batch)
                total_train_loss += loss_value.item()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.model.optimizer.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

            # validation
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_loss = train_loss(val_batch)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")



    def writeTopN(self, queries, collection, q_map:dict, d_map:dict, run_name:str, output_path:str, top_n:int = 100):
        similarities = cosine_similarity(queries, collection)
        with open(output_path, "w", newline = '') as f:
            writer = csv.writer(f, delimiter='\t')
            for i in range(len(queries)):
                top_indices = np.argsort(similarities[i])[::-1][:top_n]
                for rank, j in enumerate(top_indices):
                    writer.writerow([q_map[i], 'Q0', d_map[j], rank+1, similarities[i][j], f"{run_name}"])
                