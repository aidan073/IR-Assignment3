from dataProcessor import DataProcessor
from customBiencoder import CustomBiencoder
from customCrossencoder import CustomCrossencoder
import csv

# data processing
data = DataProcessor("data/topics_1.json", "data/Answers.json", "data/qrel_1.tsv")
topics, topic_batch, topic_map = data.getTopics(get_batch=True, get_map=True)
collection, collection_batch, collection_map = data.getCollection(get_batch=True, get_map=True)

# bi-encoder model setup
encoder = CustomBiencoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = encoder.getModel()

# run
topic_embeddings = encoder.getEmbeddings(topic_batch)
collection_embeddings = encoder.getEmbeddings(collection_batch)

# fine-tuning
qrel = data.getQrel()
samples = data.formatSamples(topics, collection, qrel)
train_data, test_data, val_data = data.test_train_split(samples)

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