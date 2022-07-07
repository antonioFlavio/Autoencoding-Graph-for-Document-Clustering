"""
KCG is Keyword Correlation Graph
"""

from src.modelling.NMF_keyword_extraction import extract_top_keywords, THE_DUMMY_NODE
from src.utils.text import split_document
from src.processing.edge_weighting import sentence_similarity_edge
from src.modelling.SBERT_transformer import get_sentence_transformer
from src.utils.datasets import fetch_dataset
import paths
import numpy as np
import pickle
import os
import time
from pathlib import Path
from src.utils.text import preprocess
from src.utils.datasets import name_of_dataset


def create_kcg(documents, big_graph, dataset_name=None):
    """

    :param documents: list of documents, each document is taken as a string

    :return: nodes (a list of nodes including{'keyword', 'feature'});
             node['feature'] is the average of its sentences' SBERT embeddings,
             and adjacency (a 2d numpy array containing weights of the graph's adjacency);
             each weight is a float in the range [-1, 1]) computed based on sentence set similarity,
             and doc_to_node_mapping which is a mapping from documnet index to its related nodes
    """
    sentence_transformer = get_sentence_transformer()
    print("documents count: {}".format(str(len(documents))))
    documents_sentences = [split_document(doc) for doc in documents]

    doc_to_node_mapping = [[] for _ in range(len(documents_sentences))]
    # doc_to_node_mapping[i] is a list containing the indexes of the nodes related to the i-th document

    start_timer_top_keywords = time.time()
    keyword_sents = extract_top_keywords(documents_sentences, big_graph, dataset_name=dataset_name)
    end_timer_top_keywords = time.time()
    paths.time_convert("extract_top_keywords", end_timer_top_keywords - start_timer_top_keywords)

    if THE_DUMMY_NODE in keyword_sents:
        del keyword_sents[THE_DUMMY_NODE]  # not considering the dummy node in the graph

    nodes = []  # a list of {'keyword', 'feature'}
    adjacency = np.zeros([len(keyword_sents), len(keyword_sents)], dtype=float)

    print("Model loaded: {} -- Max seq lenght: {}".format(sentence_transformer._get_name(), sentence_transformer.max_seq_length))
    
    start_timer_kcg = time.time()
    for node_idx, keyword in enumerate(keyword_sents):
        print('node {}/{}'.format(node_idx, len(keyword_sents)))
        sentences_idx_tuple = keyword_sents[keyword]
        # print("Encoding {} docs".format(len(documents_sentences)))
        embeddings_list = sentence_transformer.encode(
            [documents_sentences[doc_idx][sent_idx] for doc_idx, sent_idx in sentences_idx_tuple])
        # print("Encoding finished")
        average_embeddings = np.sum(embeddings_list, axis=0) / len(sentences_idx_tuple)

        # node feature
        nodes.append({'keyword': keyword, 'feature': average_embeddings})

        # node adjacency vector
        for other_node_idx, other_keyword in enumerate(keyword_sents):
            # print('--inner loop node {}/{}'.format(other_node_idx, len(keyword_sents)))
            other_sentences_idx_tuple = keyword_sents[other_keyword]
            other_embeddings_list = sentence_transformer.encode(
                [documents_sentences[doc_idx][sent_idx] for doc_idx, sent_idx in other_sentences_idx_tuple])
            edge_weight = sentence_similarity_edge(embeddings_list, other_embeddings_list)
            adjacency[node_idx, other_node_idx] = edge_weight

        for doc_idx, _ in sentences_idx_tuple:
            doc_to_node_mapping[doc_idx].append(node_idx)

    end_timer_kcg = time.time()
    paths.time_convert("kcg created", end_timer_kcg - start_timer_kcg)

    print('nodes\' features and adjacency vector are computed')

    print('KCG is created')
    return nodes, adjacency, doc_to_node_mapping


def get_documents_kcg(dataset_path, big_graph):
    """

    :param dataset_path: paths.reuters_dataset or paths.the20news_dataset
    :return: nodes, adjacency, doc_to_node_mapping; same as create_kcg function,
             and documents_labels; a list of each document's label
    """
    dataset_name = name_of_dataset(dataset_path)
    if big_graph:
        graph_file_path = paths.models + 'keyword_correlation_graph/big_' + dataset_name + '.pkl'
    else:
        graph_file_path = paths.models + 'keyword_correlation_graph/' + dataset_name + '.pkl'
    data = fetch_dataset(dataset_path)

    obj_graph_file_path = Path(graph_file_path)

    if not obj_graph_file_path.parent.exists():
        obj_graph_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        nodes, adjacency, doc_to_node_mapping = pickle.load(open(graph_file_path, 'rb'))
        documents_labels = data[:, 0]
        print('KCG is loaded')
    except FileNotFoundError:
        documents_labels = data[:, 0]
        documents = data[:, 1]
        documents = [preprocess(doc) for doc in documents]
        nodes, adjacency, doc_to_node_mapping = create_kcg(documents, big_graph, dataset_name)
        pickle.dump((nodes, adjacency, doc_to_node_mapping), open(graph_file_path, 'wb'))

    return nodes, adjacency, doc_to_node_mapping, documents_labels
