import csv
import json
from bs4 import BeautifulSoup

class DataProcessor():
    def __init__(self, topics_path:str, collection_path = None, qrel_path = None) -> None:
        with open(topics_path, "r", encoding="utf-8") as f1:
            self.topics_object = json.load(f1)
        with open(collection_path, "r", encoding="utf-8") as f2:
            self.collection_object = json.load(f2)
        self.qrel_path = qrel_path

    def parseText(self, text:str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def getTopics(self) -> dict:
        topic_dict = {}
        for topic in self.topics_object:
            topic_dict[topic["Id"]] = self.parseText(topic["Title"]) + " " + self.parseText(topic["Body"])
        return topic_dict

    def getCollection(self) -> dict:
        collection_dict = {}
        for doc in self.collection_object:
            collection_dict[doc["Id"]] = self.parseText(doc["Text"])
        return collection_dict

    def getQrel(self) -> dict:
        with open(self.qrel_path, "r", encoding="utf-8") as f3:
            qrel_reader = csv.reader(f3, delimiter='\t')
            qrel_dict = {}
            for row in qrel_reader:
                q_id = row[0]
                d_id = row[2]
                score = int(row[3])
                if q_id in qrel_dict:
                    qrel_dict[q_id].append((d_id, score))
                else:
                    qrel_dict[q_id] = [(d_id,score)]
            return qrel_dict

    def formatSamples(self, topics:dict, collection:dict, qrel:dict) -> list:
        samples = []
        for q_id,value in qrel.items():
            for d_id, score in value:
                sample = [topics[q_id], collection[d_id], score]
                samples.append(sample)
        return samples

    def tokenize(self, text):
        pass