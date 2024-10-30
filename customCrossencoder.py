from sentence_transformers import InputExample, CrossEncoder, util
from sentence_transformers.evaluation import BinaryClassificationEvaluator, CESoftmaxAccuracyEvaluator

import torch
import csv
from torch.utils.data import DataLoader
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
    def rerank(self, dict, topics, collection) -> None:
        with open('result_ce_1.tsv', 'w', newline='') as cetsv:
            ce_writer = csv.writer(cetsv, delimiter='\t')
            for q_id, answers in dict.items():
                unranked_dict = {}
                for a_id in answers:
                    score = self.model.predict([(topics[q_id], collection[a_id])])
                    unranked_dict[a_id] = score[0]
                ranked_dict = dict(sorted(unranked_dict.items(), key=lambda item: item[1], reverse=True))
                rank = 1
                for key, value in ranked_dict.items():
                    ce_writer.writerow([q_id, 'Q0', key, rank, value, "ce1"])
                    rank += 1
    def fine_tune(self, train_data, val_data, test_data, epochs=1, batch_size=16, warmup_steps=100):
        train_examples = [InputExample(texts=[ex[0], ex[1]], label=ex[2]) for ex in train_data]
        val_examples = [InputExample(texts=[ex[0], ex[1]], label=ex[2]) for ex in val_data]
        test_examples = [InputExample(texts=[ex[0], ex[1]], label=ex[2]) for ex in test_data]
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(val_examples, name="validation")

        self.model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path="fine_tuned_crossencoder"
        )

        evaluator = BinaryClassificationEvaluator.from_input_examples(val_examples, name="validation")
        test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_examples, name="test")

