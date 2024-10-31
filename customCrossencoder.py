from sentence_transformers import InputExample, CrossEncoder, util
from sentence_transformers.evaluation import BinaryClassificationEvaluator
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
        evaluator = BinaryClassificationEvaluator.from_input_examples(val_data, name="validation")

        self.model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            output_path="ftcrossencoder" 
        )

        # Evaluate on the test set after fine-tuning
        test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_data, name="test")
        test_evaluator(self.model)

# Data Loading
data = DataProcessor("data/topics_1.json", "data/Answers.json", "data/qrel_1.tsv")
topics, topic_batch, topic_map = data.getTopics(get_batch=True, get_map=True)
collection, collection_batch, collection_map = data.getCollection(get_batch=True, get_map=True)

# Initialize CrossEncoder and Data
crossencoder = CustomCrossencoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tsvdict1 = data.readTSV('test.tsv') #result_bi_1.tsv
tsvdict2 = data.readTSV('test.tsv') #result_bi_2.tsv
crossencoder.rerank(tsvdict1, topics, collection, "result_ce_1.tsv")
crossencoder.rerank(tsvdict2, topics, collection, "result_ce_2.tsv")


data = DataProcessor("data/topics_2.json", "data/Answers.json", "data/qrel_1.tsv")
topics, topic_batch, topic_map = data.getTopics(get_batch=True, get_map=True)
collection, collection_batch, collection_map = data.getCollection(get_batch=True, get_map=True)

qrel = data.getQrel()
samples = data.formatSamples(topics, collection, qrel)
train_data, test_data, val_data = data.test_train_split(samples)

crossencoder.fine_tune(train_data, val_data, test_data) #

fine_tuned_crossencoder = CustomCrossencoder("ftcrossencoder") #
tsvdict1 = data.readTSV('test.tsv') #result_bi_ft_1.tsv
tsvdict2 = data.readTSV('test.tsv') #result_bi_ft_2.tsv
fine_tuned_crossencoder.rerank(tsvdict1, topics, collection, "result_ce_ft_1.tsv")
fine_tuned_crossencoder.rerank(tsvdict2, topics, collection, "result_ce_ft_2.tsv")
