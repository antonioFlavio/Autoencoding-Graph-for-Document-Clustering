"""
Data cleansing based on the details in the paper
"Determining Gains Acquired from Word Embedding Quantitatively Using Discrete Distribution Clustering"
"""
import sys

from tqdm import tqdm 
sys.path.append('./')
# import paths
import os
import re
import csv

try:
    from src.paths import *
except:
    from paths import *
import re

def __prepare_the20nes():
    ds = open(the20news_dataset, mode='w')  # saving cleaned database
    ds_writer = csv.writer(ds, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ds_writer.writerow(['category', 'document'])  # column names

    for cat in os.listdir(the20news_original_dataset):
        cat_adr = the20news_original_dataset + cat
        if os.path.isdir(cat_adr):
            for doc_file in os.listdir(cat_adr):
                doc_adr = cat_adr + '/' + doc_file
                with open(doc_adr, encoding='utf-8', errors='ignore') as doc:
                    lines = doc.readlines()
                    if len(lines) > 3:
                        doc = ''.join(lines[3:])  # remove From and Subject
                        doc = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|'
                                     r'\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!'
                                     r'()\[\]{};:\'".,<>?«»“”‘’]))', '', doc)  # remove URls
                        doc = re.sub(r'\S*@\S*\s?', '', doc)  # remove email addresses
                        doc = re.sub(r'(\t|\n| )+', ' ', doc).strip()

                        if len(doc.split()) >= 10:  # delete documents with less than ten words
                            ds_writer.writerow([cat, doc])

    ds.close()


def __read_dataset():  # 18692 documents in 20 categories
    with open(the20news_dataset) as ds:
        reader = csv.reader(ds, delimiter=',')
        headers = next(reader)
        for row in reader:
            print('Category:  ', row[0])
            print('Document:  ', row[1])

__prepare_the20nes()