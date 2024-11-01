from dataProcessor import DataProcessor
from customBiencoder import CustomBiencoder
from customCrossencoder import CustomCrossencoder
import argparse

parser = argparse.ArgumentParser(description="Process paths for topics, collection, fine-tuned model paths, result file output paths, optional qrel and embeddings.")

# positional arguments
parser.add_argument("topic_path", type=str, help="Path to the topic file")
parser.add_argument("collection_path", type=str, help="Path to the collection file")
parser.add_argument("-f", "--fine_tuned_paths", nargs=2, metavar=("BI_ENCODER_FT_PATH", "CE_FT_PATH"),help="Path to fine-tuned models")
parser.add_argument("-o", "--output_paths", nargs=4, required=True, metavar=("BI_OUTPUT_PATH", "BI_OUTPUT_PATH_FT", "CE_OUTPUT_PATH", "CE_OUTPUT_PATH_FT"),help="Desired output paths")  
parser.add_argument("-q", "--qrel_path", type=str, help="Path to qrel file")
parser.add_argument("-L", "--load_paths", nargs=2, metavar=("TOPIC_EMBEDDINGS_PATH", "ANSWER_EMBEDDINGS_PATH"),help="Paths containing pre-encoded topic/answer embeddings for bi-encoder")

args = parser.parse_args()

# checking command line args
topic_path = args.topic_path
collection_path = args.collection_path
bi_output_path, bi_output_path_ft, ce_output_path, ce_output_path_ft = args.output_paths
fine_tuned_paths = None
qrel_path = None
load_paths = None 
if args.fine_tuned_paths:
    ft_bi_path, ft_ce_path = args.fine_tuned_paths
if args.qrel_path:
    qrel_path = args.qrel_path
if args.load_paths:
    topic_emb_path, collection_emb_path = args.load_paths

# data processing
data = DataProcessor(topic_path, collection_path, qrel_path)
topics, topic_batch, topic_map = data.getTopics(get_batch=True, get_map=True)
collection, collection_batch, collection_map = data.getCollection(get_batch=True, get_map=True)
print("Data processing complete\n")

# bi-encoder model setup
encoder = CustomBiencoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = encoder.getModel()

# run base bi-encoder
if load_paths:
    topic_embeddings = encoder.loadEmbeddings(topic_emb_path)
    collection_embeddings = encoder.loadEmbeddings(collection_emb_path)
else:
    topic_embeddings = encoder.getEmbeddings(topic_batch)
    collection_embeddings = encoder.getEmbeddings(collection_batch)
encoder.writeTopN(topic_embeddings, collection_embeddings, "bi_encoder", bi_output_path, q_map=topic_map, d_map=collection_map)

print("Bi-encoder base results obtained\n")

# run base crossencoder
crossencoder = CustomCrossencoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tsvdict = data.readTSV(bi_output_path)
crossencoder.rerank(tsvdict, topics, collection, ce_output_path)
print("Crossencoder base results obtained\n")

# if qrel_path command line arg provided
if qrel_path:
    qrel = data.getQrel()
    samples = data.formatSamples(topics, collection, qrel)
    train_data, test_data, val_data = data.test_train_split(samples)
    # if fine-tuned models not passed in command line args
    if not fine_tuned_paths:
        encoder.fineTune(train_data, val_data, 3)
        model.save("model/fine-tuned-be")
        crossencoder.fine_tune(train_data, val_data, test_data)
        print("Fine-tuning complete\n")
    # if fine-tuned models passed in command line args
    else:
        encoder.loadModel(ft_bi_path)
        fine_tuned_crossencoder = CustomCrossencoder(ft_ce_path)
    # get results for fine-tuned models
    test_topic_batch, test_collection_batch, t_qmap, t_dmap = data.getSubsetBatches(test_data)
    test_topic_embs = encoder.getEmbeddings(test_topic_batch)
    test_collection_embs = encoder.getEmbeddings(test_collection_batch)
    data.genQrel(test_data)
    encoder.writeTopN(test_topic_embs, test_collection_embs, "bi_encoder_fine-tuned", bi_output_path_ft, q_map=t_qmap, d_map=t_dmap)
    print("Bi-encoder fine-tuned results obtained\n")
    ft_tsvdict = data.readTSV(bi_output_path_ft)
    fine_tuned_crossencoder.rerank(ft_tsvdict, topics, collection, ce_output_path_ft)
    print("Crossencoder fine-tuned results obtained\n")
# if no qrel_path command line arg provided, assume fine-tuned models were provided
else:
    encoder.loadModel(ft_bi_path)
    fine_tuned_crossencoder = CustomCrossencoder(ft_ce_path)
    topic_embeddings = encoder.getEmbeddings(topic_batch)
    collection_embeddings = encoder.getEmbeddings(collection_batch)
    encoder.writeTopN(topic_embeddings, collection_embeddings, "bi_encoder_fine-tuned", bi_output_path_ft, q_map=topic_map, d_map=collection_map)
    print("Bi-encoder fine-tuned results obtained\n")
    ft_tsvdict = data.readTSV(bi_output_path_ft)
    fine_tuned_crossencoder.rerank(ft_tsvdict, topics, collection, ce_output_path_ft)
    print("Crossencoder fine-tuned results obtained\n")

