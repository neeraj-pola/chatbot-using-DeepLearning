import random
import json
import pickle
import numpy as np

import nltk # natural language tool kit
from nltk.stem import WordNetLemmatizer
# WordNetLemmatizer reduces the word to its stem.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())
# reads the files from intents and loads it into intents variable with json object

words = []  # contains each word of a given sentence seperated with comma
classes = [] # contains tags
documents = [] # contains words with related tags
ignore_letters = ["?", ",", ".", "!"]


for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern) # breaks a sentence into words seperated by comma
        words.extend(word_list)
        documents.append((word_list, intent["tags"]))
        if intent["tags"] not in classes:
            classes.append(intent['tags'])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

# sets and sorts both words list and classes list
words = sorted(set(words))
classes = sorted(set(classes))

# save and add them to pickle files by using the term dump.
pickle.dump(words, open("words.pkl", 'wb'))
pickle.dump(classes, open("classes.pkl", 'wb'))

# we convert the words into numbers for data processing(o's and 1's)

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    # we then add all words from document to word_patterns and lemmatize it.
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        # append 1 if word is in word_patterns else 0 in bag
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# we randomize training data for more flexibility
random.shuffle(training)
# change the training array into numpy array
training = np.array(training)

# first part of training data to train_x and second part to train_y
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# we then create a sequential function form class tensorflow.keras.models with model as an object
model = Sequential()
# shape of layer will be equal to sizeof train_x and activation layer is rectified linear unit.
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# softmax scales the results of the output so they all can be added upto be 1.
model.add(Dense(len(train_y[0]), activation='softmax'))

# create an sgd object from class tensorflow.keras.optimizers with adding learning rate..
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile the model to get matrices of accuracy.
# used categorical_crossentropy for loss.
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

# we create and file chatbotmodel.h5 with fit model and attributes and start training the data.
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
# prints done if training completed.
print('Done')

