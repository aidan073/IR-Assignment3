from dataProcessor import DataProcessor
from customBiencoder import CustomBiencoder

# data processing
data = DataProcessor("data/topics_1.json", "data/Answers.json", "data/qrel_1.tsv")
topics = data.getTopics()
collection = data.getCollection()
qrel = data.getQrel()
samples = data.formatSamples(topics, collection, qrel)

# bi-encoder model setup
encoder = CustomBiencoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = encoder.getModel()