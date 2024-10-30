from dataProcessor import DataProcessor
from customBiencoder import CustomBiencoder
from customCrossencoder import CustomCrossencoder
import argparse
import csv

#parser = argparse.ArgumentParser(description="Process paths for topics, collections, qrels, and answers.")

# positional arguments
# parser.add_argument("topic_path", type=str, help="Path to the topic file")
# parser.add_argument("collection_path", type=str, help="Path to the collection file")
# parser.add_argument("-f", "--qrel_path", type=str, help="Path to the qrel file")
# parser.add_argument("-L", "--load_paths", nargs=2, metavar=("TOPICS_LOAD_PATH", "ANSWERS_LOAD_PATH"),
#                     help="Paths containing pre-encoded topic/answer embeddings")

# arguments = parser.parse_args()

# data processing
data = DataProcessor("data/topics_1.json", "data/Answers.json", "data/qrel_1.tsv")
#data = DataProcessor("data/topics_1_mini.json", "data/Answers_mini.json")
topics, topic_batch, topic_map = data.getTopics(get_batch=True, get_map=True)
collection, collection_batch, collection_map = data.getCollection(get_batch=True, get_map=True)

# bi-encoder model setup
encoder = CustomBiencoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = encoder.getModel()

# run
# outfile_name = "topics"
# topic_embeddings = encoder.getEmbeddings(topic_batch, outfile_name)
# outfile_name = "collection"
# collection_embeddings = encoder.getEmbeddings(collection_batch, outfile_name)
topic_embeddings = encoder.loadEmbeddings("topics.npy")
collection_embeddings = encoder.loadEmbeddings("collection.npy")
encoder.writeTopN(topic_embeddings, collection_embeddings, topic_map, collection_map, "bi_encoder", "test.tsv")

# fine-tuning
qrel = data.getQrel()
samples = data.formatSamples(topics, collection, qrel)
train_data, test_data, val_data = data.test_train_split(samples)

# cross-encoder model setup
crossencoder = CustomCrossencoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
crossmodel = crossencoder.getModel()
testdict = data.readTSV('data/result_bm25_1.tsv')

#crossmodel.rerank(testdict, topics, collection)