#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:03:33 2021

@author: sashaqanderson
"""
from typing import Dict, List
import json
import pandas as pd
import requests

URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16

def chunks(lst, chunk_size=MAX_BATCH_SIZE):
    """Splits a longer list to respect batch size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

df = pd.read_csv('/Users/sashaqanderson/Desktop/contextualized-topic-models-master/abstracts.csv')
# print(df.columns)

df = df[['Unnamed: 0', 'Article name', 'Title', 'Text']]
df.columns = ['doc_id', 'link', 'title', 'abstract']
df['doc_id'] = df['doc_id'].astype(str)


SAMPLE_PAPERS = []
for i in range(len(df)):
    SAMPLE_PAPERS.append({"paper_id": df.doc_id[i], "title": df.title[i],
                        "abstract": df.abstract[i]})

# SAMPLE_PAPERS = [
#     {
#         "paper_id": "A",
#         "title": "Angiotensin-converting enzyme 2 is a functional receptor for the SARS coronavirus",
#         "abstract": "Spike (S) proteins of coronaviruses ...",
#     },
#     {
#         "paper_id": "B",
#         "title": "Hospital outbreak of Middle East respiratory syndrome coronavirus",
#         "abstract": "Between April 1 and May 23, 2013, a total of 23 cases of MERS-CoV ...",
#     },
# ]

def embed(papers):
    embeddings_by_paper_id: Dict[str, List[float]] = {}

    for chunk in chunks(papers):
        # Allow Python requests to convert the data above to JSON
        response = requests.post(URL, json=chunk)

        if response.status_code != 200:
            raise RuntimeError("Sorry, something went wrong, please try later!")

        for paper in response.json()["preds"]:
            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]

    return embeddings_by_paper_id


if __name__ == "__main__":
    all_embeddings = embed(SAMPLE_PAPERS)
    # Prints { 'A': [4.089589595794678, ...], 'B': [-0.15814849734306335, ...] }
    print(all_embeddings)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#OLDER METHOD:
    
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
# model = AutoModel.from_pretrained('allenai/specter')

# df = pd.read_csv('/Users/sashaqanderson/Desktop/contextualized-topic-models-master/abstracts.csv')
# # print(df.columns)

# df = df[['Unnamed: 0', 'Article name', 'Title', 'Text']]
# df.columns = ['doc_id', 'link', 'title', 'abstract']
# print(df.head)

# text_list = []
# for i in range(len(df)):
#     text_list.append({"title": df.title[i],
#                         "abstract": df.abstract[i]})
  
# # concatenate title and abstract
# title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in text_list]
# # print(title_abs[0])

# # preprocess the input
# inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
# result = model(**inputs)

# # take the first token in the batch as the embedding
# embeddings = result.last_hidden_state[:, 0, :]
# print(embeddings.shape)