import csv
import json
import random
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
            temp = []
            for tuple in items:
                temp.append(InputExample(guid = f"{q_id},{tuple[0]}", texts=[topics[q_id], collection[tuple[0]]], label=1.0 if tuple[1] >= 1 else 0.0))
            samples.append(temp)
        return samples # [[*all inputexamples for a qid], ...]

    # given the testing set, find its corresponding embeddings
    def getSubsetBatches(self, test_examples):
        seen_qids = set() # prevent adding duplicate sentences to query batch
        seen_dids = set() # prevent adding duplicate sentences to document batch
        q_batch = []
        d_batch = []
        qmap = {}
        dmap = {}
        current_qid_index = 0
        current_did_index = 0
        for example in test_examples:
            example_ids = example.guid.split(",")
            q_id = int(example_ids[0])
            d_id = int(example_ids[1])
            if q_id not in seen_qids: # if qid has not been seen
                seen_qids.add(q_id) # qid has now been seen
                q_batch.append(example.texts[0]) # add query text to batch
                qmap[current_qid_index] = q_id # map batch index to qid for retrieval later on
                current_qid_index+=1 
            if d_id not in seen_dids: # if did has not been seen
                seen_dids.add(d_id) # did has now been seen
                d_batch.append(example.texts[1]) # add document text to batch
                dmap[current_did_index] = d_id # map batch index to did for retrieval later on
                current_did_index+=1
        return q_batch, d_batch, qmap, dmap
    
    # need to make a new qrel for the test set
    def genQrel(self, test_examples):
        with open("test_qrel.tsv", "w", newline = '') as f:
            writer = csv.writer(f, delimiter='\t')
            for example in test_examples:
                example_ids = example.guid.split(",")
                q_id = example_ids[0]
                d_id = example_ids[1]
                score = int(example.label)
                writer.writerow([q_id, 0, d_id, score])


    def test_train_split(self, samples):
        if len(samples) < 20: # will result in empty splits
            raise ValueError("Sample size must be >= 20")

        # shuffle data
        random.seed(73)
        random.shuffle(samples)

        # calculate splits for 90/5/5 and organize samples into train-test-val
        train_end = round(0.9 * len(samples))
        val_end = train_end + round(0.05 * len(samples))
        train_data = samples[:train_end]
        test_data = samples[val_end:]
        val_data = samples[train_end:val_end]

        # unpack samples, which were grouped by qid, into individiual samples
        train_data = [example for group in train_data for example in group]
        test_data = [example for group in test_data for example in group]
        val_data = [example for group in val_data for example in group]

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