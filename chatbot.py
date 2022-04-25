import pandas as pd
import numpy as np
import nltk
import json
import tensorflow as tf
from tensorflow import keras
import pickle
nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation 
from keras.optimizers import gradient_descent_v2

words = []
classes = []
documents = []
ignore_words = ["?","!"]
data_file = open("job_intents.json",encoding = "utf-8").read()
intents = json.loads(data_file)

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
            
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(classes)))

print(len(documents),"documents")

print(len(classes),"classes",classes)

print(len(words),"unique lemmatized words", words)

pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))
            
#Initialize the training data.
training = []
output_empty = [0]*len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

for w in words:
    bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    
random.shuffle(training)
training = np.array(training)

#Create train and Test Split.
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
print(len(train_x[0]))

# Create model with 3 layers. 1st layer has 128 neurons, 2nd layer has 64 neurons, 3rd layer has number of neurons equal to the no. of intents to predict output intent with softmax function
model = Sequential()
print("start")
model.add(Dense(128,input_shape=(14,),activation = "relu"))
print("done")
model.add(Dropout(0.5))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation = "softmax"))

# Compile the model. Use of SGD with Nesterov accelerated gradient gives good results for this model
sgd = gradient_descent_v2.SGD(learning_rate=0.01, decay=1e-6, momentum =0.9, nesterov = True)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])

# Model fitting and Saving.
hist = model.fit(np.array(train_x),np.array(train_y), epochs = 200, batch_size = len(train_x[0]), verbose = 1)
model.save("chatbot_model.h5",hist)

print("model created")