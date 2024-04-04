### Author: Riya Nakarmi ###
### College Project ###

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Load the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open('intents.json').read())

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
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
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("\n\n|============= Welcome to Indian Constitution Equiry Chatbot System! =============|")
print("|============================== Feel Free ============================|")
print("|================================== To ===============================|")
print("|=============== Ask your any query about our Indian Constitution ================|")
while True:
    message = input("\n| You: ")
    if message.lower() == "bye" or message.lower() == "goodbye":
        print("\n\n| Bot: Goodbye! Have a nice day.")
        print("\n\n|===================== The Program Ends here! =====================|\n\n")
        break
    else:
        ints = predict_class(message)
        if not ints:
            print("\n| Bot: I'm sorry, I didn't understand that. Please try again.")
        else:
            res = get_response(ints, intents)
            print("\n| Bot:", res)