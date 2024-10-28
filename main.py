from dataProcessor import DataProcessor
from customBiencoder import CustomBiencoder
from customCrossencoder import CustomCrossencoder
import csv

# data processing
data = DataProcessor("data/topics_1.json", "data/Answers.json", "data/qrel_1.tsv")
topics = data.getTopics()
collection = data.getCollection()
qrel = data.getQrel()
samples = data.formatSamples(topics, collection, qrel)

# bi-encoder model setup
encoder = CustomBiencoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = encoder.getModel()

# cross-encoder model setup
crossencoder = CustomCrossencoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
crossmodel = crossencoder.getModel()
testdict = data.readTSV('data/result_bm25_1.tsv')

with open('result_ce_1.tsv', 'w', newline='') as cetsv:
    ce_writer = csv.writer(cetsv, delimiter='\t')
    for q_id, answers in testdict.items():
        unranked_dict = {}
        for a_id in answers:
            score = crossmodel.predict([(topics[q_id], collection[a_id])])
            unranked_dict[a_id] = score[0]
        ranked_dict = dict(sorted(unranked_dict.items(), key=lambda item: item[1], reverse=True))
        rank = 1
        for key, value in ranked_dict.items():
            ce_writer.writerow([q_id, 'Q0', key, rank, value, "ce1"])
            rank += 1