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

    def getTopics(self, get_batch:bool = False, get_map:bool = False, get_Rmap:bool=False) -> dict:
        batch = []
        map = {}
        Rmap = {} # reverse of map
        topic_dict = {}
        for i, topic in enumerate(self.topics_object):
            text = self.parseText(topic["Title"]) + " " + self.parseText(topic["Body"])
            if get_batch:
                batch.append(text)
            if get_map:
                map[i] = topic['Id']
            if get_Rmap:
                Rmap[topic['Id']] = i
            topic_dict[topic["Id"]] = text

        results = [topic_dict] # for return tuple
        if get_batch:
            results.append(batch)
        if get_map:
            results.append(map)
        if get_Rmap:
            results.append(Rmap)
        return results

    def getCollection(self, get_batch:bool = False, get_map:bool = False, get_Rmap:bool = False) -> dict:
        batch = []
        map = {}
        Rmap = {}
        collection_dict = {}
        for i, doc in enumerate(self.collection_object):
            text = self.parseText(doc["Text"])
            if get_batch:
                batch.append(text)
            if get_map:
                map[i] = doc['Id']
            if get_Rmap:
                Rmap[doc['Id']] = i
            collection_dict[doc["Id"]] = text

        results = [collection_dict] # for return tuple
        if get_batch:
            results.append(batch)
        if get_map:
            results.append(map)
        if get_Rmap:
            results.append(Rmap)
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
        for q_id, items in qrel.items():
            for tuple in items:
                samples.append(InputExample(guid = f"{q_id}, {tuple[0]}", texts=[topics[q_id], collection[tuple[0]]], label=1.0 if tuple[1] >= 1 else 0.0))
        return samples

    # given the testing set, find its corresponding embeddings
    def getSubsetBatches(self, test_examples, t_Rmap, d_Rmap, t_embs, d_embs):
        qembs = {}
        dembs = {}
        qmap = {}
        dmap = {}
        for idx, example in enumerate(test_examples):
            example_ids = example.guid.split(",")
            q_id = example_ids[0]
            d_id = example_ids[1]
            if q_id not in qembs:
                qembs[q_id] = t_embs[t_Rmap[q_id]]
                qmap[idx] = q_id
            if d_id not in dembs:
                dembs[d_id] = d_embs[d_Rmap[d_id]]
                dmap[idx] = d_id
        return qembs, dembs, qmap, dmap
    
    # need to make a new qrel for the test set
    def genQrel(self, test_examples):
        with open("test_qrel.tsv", "w", newline = '') as f:
            writer = csv.writer(f, delimiter='\t')
            for example in test_examples:
                example_ids = example.guid.split(",")
                q_id = example_ids[0]
                d_id = example_ids[1]
                score = example.label
                writer.writerow([q_id, 0, d_id, score])


    def test_train_split(self, samples):
        # shuffle data
        samples = np.array(samples)
        np.random.seed(73)
        np.random.shuffle(samples)

        # calculate splits for 90/5/5 and organize samples into train-test-val
        train_end = round(0.9 * len(samples))
        val_end = train_end + round(0.5 * len(samples))
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