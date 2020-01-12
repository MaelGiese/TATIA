import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def data_split_label(data, label):
    # Séparation du label
    y = pd.get_dummies(data[label]).values
    return y
def clean_long_sentence(df, max_word, column_target):
    df = df[df[column_target].str.split().str.len() < max_word]
    return df

def get_clean_df(pathFile):
    df = pd.read_json(pathFile, lines = True)
    # On garde uniquement les colonnes qui nous seront utile
    df = df[['headline','is_sarcastic']]

    df['headline'] = df['headline'].apply(lambda x: x.lower())
    # On filtre les symboles spéciaux
    df['headline'] = df['headline'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    return df

# Transforme les titres en vecteurs pour que le model puisse les lire
def get_vectorize_df(df, target_column):
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(df[target_column].values)
    X = tokenizer.texts_to_sequences(df[target_column].values)
    X = pad_sequences(X)
    return X, tokenizer


# Transforme les titres en vecteurs
def get_word2vec_df(df, we_model, target_column, max_word, vector_size):
    #Word2Vec
    #vocab = we_model.wv.vocab.keys()
    vocab = we_model.keys()
    nullWord = np.zeros((vector_size),dtype=float)
    compt = 0
    X = []
    
    for index, row in df.iterrows():
        vectors = []
        sentence = row[target_column]
        
        for i in range(max_word):
            if i < len(sentence):
                w = sentence[i]
                if w in vocab:
                    # Word2Vec
                    #for n in we_model.wv.word_vec(w):
                    for n in we_model[w]:
                        vectors.append(n+10)
                        if(n+2 < -1):
                            print(n)
                else:
                    compt = compt+1
                    for n in nullWord:
                        vectors.append(n)
            else:
                    for n in nullWord:
                        vectors.append(n)
            
        X.append(vectors)
    print(compt)
    return np.array(X)







