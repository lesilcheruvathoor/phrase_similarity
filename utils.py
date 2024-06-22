import gensim
import csv
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

f_path = 'C:/Users/lesil/PycharmProjects/phrase_similarity/GoogleNews-vectors-negative300.bin.gz'

def file_generated_for_vectors():
    # load the word embeddings from location
    wv = KeyedVectors.load_word2vec_format(f_path, binary=True, limit=1000000)
    wv.save("vectors.csv")
    return wv

def load_vectors(filepath):
    loaded_emb ={}
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            word = row[0]
            vector = np.array(row[1:],dtype=np.float32)
            loaded_emb[word] = vector
        return loaded_emb

def load_phrases(filepath):
    return pd.read_csv(filepath, encoding='ISO-8859-1')

def save_file(dist,filepath):
    df = pd.DataFrame(dist)
    df.to_csv(filepath, index=False)
    return

def read_input_file():
    with open("input_phrase.txt", 'r') as file:
        data = file.read()
        print(data)
        return data
