import pandas as pd
import numpy as np
import model as mo
import Word_embedding as we
import Data_pre_treatment as dp
import pretrained_word_embedding as pwe

df = dp.get_clean_df('Datasets\Sarcasm_Headlines_Dataset_v2.json')

# supprime les titres qui contiennent plus de 10 mots
#df = dp.clean_long_sentence(df, 10, 'headline')

y = dp.data_split_label(df,'is_sarcastic')
X, tokenizer = dp.get_vectorize_df(df, 'headline')


# nombre de mot maximum parmis tout les titres 
#max_word = df['headline'].str.split().str.len().max()

# création d'un model word2vec a partir du dataset
#we_model = we.get_word_embedding_model('Datasets\Sarcasm_Headlines_Dataset_v2.json')

# pretrained word embedding dictionnary
#we_model = pwe.get_word_embedding_dict("glove.6B\glove.6B.100d.txt")
#X = dp.get_word2vec_df(df, we_model, 'headline',max_word, 100)

# train the model
#model = mo.get_model(X, y)


model = mo.load_model()

# Test si la phrase est sarcastique ou non
mo.is_sarcasm(model, tokenizer, 'I like trains')
mo.is_sarcasm(model, tokenizer, 'Everyone assumes I’m psychopath, except for my friends who live deep inside my head.')

