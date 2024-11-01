from sentence_transformers import InputExample, CrossEncoder, util
from dataProcessor import DataProcessor
import torch
import csv
from torch.utils.data import DataLoader

class CustomCrossencoder():
    def __init__(self, model_name: str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
    def getModel(self):
        return self.model
    def setModel(self, model) -> None:
        self.model = model
    def cosineSim(self, emb1, emb2) -> torch.Tensor:
        return util.cos_sim(emb1, emb2)
    def rerank(self, testdict, topics, collection, output_file) -> None:
        with open(output_file, 'w', newline='') as cetsv:
            ce_writer = csv.writer(cetsv, delimiter='\t')
            for q_id, answers in testdict.items():
                unranked_dict = {}
                for a_id in answers:
                    score = self.model.predict([(topics[q_id], collection[a_id])])
                    unranked_dict[a_id] = score[0]
                ranked_dict = dict(sorted(unranked_dict.items(), key=lambda item: item[1], reverse=True))
                rank = 1
                for key, value in ranked_dict.items():
                    ce_writer.writerow([q_id, 'Q0', key, rank, value+20, "ce1"])
                    rank += 1
    def fine_tune(self, train_data, val_data, test_data, epochs=1, batch_size=16, warmup_steps=100):
        # Prepare InputExample format for train, validation, and test data
        if not isinstance(train_data[0], InputExample):
            train_data = [InputExample(texts=[ex[0], ex[1]], label=ex[2]) for ex in train_data]
        if not isinstance(val_data[0], InputExample):
            val_data = [InputExample(texts=[ex[0], ex[1]], label=ex[2]) for ex in val_data]
        if not isinstance(test_data[0], InputExample):
            test_data = [InputExample(texts=[ex[0], ex[1]], label=ex[2]) for ex in test_data]

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        def custom_evaluate(data, name):
            correct_predictions = 0
            for example in data:
                prediction_score = self.model.predict([(example.texts[0], example.texts[1])])[0]
                predicted_label = 1 if prediction_score >= 0.5 else 0
                if predicted_label == example.label:
                    correct_predictions += 1
            accuracy = correct_predictions / len(data)
            return accuracy

        # Fine-tuning loop
        self.model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps
        )

        # Save the model and tokenizer after fine-tuning
        model_path = "model/ftcrossencoder"
        self.model.save_pretrained(model_path)
        self.model.tokenizer.save_pretrained(model_path)

        # Evaluate on validation and test sets
        custom_evaluate(val_data, name="Validation")
        custom_evaluate(test_data, name="Test")