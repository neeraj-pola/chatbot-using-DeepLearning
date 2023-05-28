import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

# the trained model which is loaded in chatbotmodel.h5 is now to be transferred into this file.
from tensorflow.keras.models import load_model

lemmitizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

# we collect the info from all the pickle files
words = pickle.load(open("words.pkl", 'rb'))
classes = pickle.load(open("classes.pkl", 'rb'))
# model object is created from training data file.
model = load_model("chatbotmodel.h5")


# create a function for tokenizing the sentence and returning it.
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmitizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# create a bag of words using the sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    # create a bag of 0's with length of words
    bag = [0] * len(words)
    # check whether if both input sentence words and existing words in intents are equal
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                # if equal than append 1 to the index position of that particular word
                bag[i] = 1
    return np.array(bag)


# predict sentence with given sentence.
def predict_class(sentence):
    # create a bag of words using previous function
    bow = bag_of_words(sentence)
    # get result with using predict from the model.
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    # get the result after checking the condition of error threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tags'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("start")

while True:
    message = input('')
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)