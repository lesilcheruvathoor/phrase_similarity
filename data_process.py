import numpy as nm
from sklearn.metrics.pairwise import cosine_distances


def get_phrase_vector(phrase,word_vect):
    word = phrase.split()
    wv = [word_vect[w] for w in word if w in word_vect]
    print(wv)
    if not wv:
        return nm.zeros(word_vect.vector_size)
    return nm.mean(wv, axis=0)

def calc_distances(df):
    phrase_vectors = nm.stack(df['vector'].values)
    dist = cosine_distances(phrase_vectors)
    return dist

def identify_phrase(inp, wd, df_phrase):
    inp_vector = get_phrase_vector(inp,wd)
    phrase_vect = nm.stack(df_phrase["vector"].values)
    dist = cosine_distances([inp_vector],phrase_vect)
    closest_idx = nm.argmin(dist)
    closest_phrase = df_phrase.iloc[closest_idx]['Phrases']
    return closest_phrase, dist[0,closest_idx]