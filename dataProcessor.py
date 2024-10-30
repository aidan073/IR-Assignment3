import csv
import json
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
from sentence_transformers import InputExample

class DataProcessor():
    def __init__(self, topics_path:str, collection_path:str, qrel_path = None) -> None:
        with open(topics_path, "r", encoding="utf-8") as f1:
            self.topics_object = json.load(f1)
        with open(collection_path, "r", encoding="utf-8") as f2:
            self.collection_object = json.load(f2)
        self.qrel_path = qrel_path

    def parseText(self, text:str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def getTopics(self, get_batch:bool = False, get_map:bool = False) -> dict:
        batch = []
        map = {}
        topic_dict = {}
        for i, topic in enumerate(self.topics_object):
            text = self.parseText(topic["Title"]) + " " + self.parseText(topic["Body"])
            if get_batch:
                batch.append(text)
            if get_map:
                map[i] = topic['Id']
            topic_dict[topic["Id"]] = text

        results = [topic_dict] # for return tuple
        if get_batch:
            results.append(batch)
        if get_map:
            results.append(map)
        return results

    def getCollection(self, get_batch:bool = False, get_map:bool = False) -> dict:
        batch = []
        map = {}
        collection_dict = {}
        for i, doc in enumerate(self.collection_object):
            text = self.parseText(doc["Text"])
            if get_batch:
                batch.append(text)
            if get_map:
                map[i] = doc['Id']
            collection_dict[doc["Id"]] = text

        results = [collection_dict] # for return tuple
        if get_batch:
            results.append(batch)
        if get_map:
            results.append(map)
        return results

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
                samples.append(InputExample(texts=[topics[q_id], collection[d_id]], label=1 if score >= 1 else 0))
        return samples

    def test_train_split(self, samples):
        # shuffle data
        samples = np.array(samples)
        np.random.seed(73)
        np.random.shuffle(samples)

        # calculate splits for 90/5/5 and organize samples into train-test-val
        train_end = int(0.9 * len(samples))
        val_end = train_end + int(0.05 * len(samples))
        train_data = samples[:train_end]
        test_data = samples[val_end:]
        val_data = samples[train_end:val_end]

        return train_data, test_data, val_data

    def readTSV(self, result_path:str) -> dict:
        result = defaultdict(list)

        with open(result_path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                qid = row[0]
                value = row[2]
                result[qid].append(value)
        return dict(result)