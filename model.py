import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import model_from_json


def is_sarcasm(model, tokenizer, sentence):
    headline = [sentence]
    headline = tokenizer.texts_to_sequences(headline)
    headline = pad_sequences(headline, maxlen=76, dtype='int32', value=0)
    
    sentiment = model.predict(headline,batch_size=1,verbose = 2)[0]
    print()
    print(sentence)
    if(np.argmax(sentiment) == 0):
        print("Non-sarcastic")
    elif (np.argmax(sentiment) == 1):
        print("Sarcasm")

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        
    model.save_weights("model.h5")
    
def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    return loaded_model
    
def get_model(X, y):
    ####################################
    # Model

    X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)
    
    max_fatures = 2000
    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

    

    SVG(model_to_dot(model).create(prog='dot', format='svg'))

    # Training
    batch_size = 32
    history = model.fit(X_train, Y_train, epochs = 25, batch_size=batch_size, verbose = 2)

    # Testing the model
    validation_size = 1500

    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_validate)):
        
        result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
       
        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1
           
        if np.argmax(Y_validate[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1



    print("Sarcasm_acc", pos_correct/pos_cnt*100, "%")
    print("Non-Sarcasm_acc", neg_correct/neg_cnt*100, "%")
    return model


    



