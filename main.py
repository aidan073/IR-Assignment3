from dataProcessor import DataProcessor

data = DataProcessor("data/topics_1.json", "data/Answers.json", "data/qrel_1.tsv")
topics = data.getTopics()
collection = data.getCollection()
qrel = data.getQrel()
data.formatSamples(topics, collection, qrel)
