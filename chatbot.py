import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pyttsx3
import speech_recognition as sr
import pyaudio
from keras.models import load_model
import json
import random
import os
import tkinter
from tkinter import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lemmetizer = WordNetLemmatizer()
intents = json.loads(open('keskustelut.json', encoding="utf-8").read())
words = pickle.load(open('sanat.pkl', 'rb'))
classes = pickle.load(open('luokat.pkl', 'rb'))
model = load_model('ChattiMalli.h5')


def kasa_sanoja(lause,words,show_details=True):
    lauseen_sanat = nltk.word_tokenize(lause)
    lauseen_sanat = [lemmetizer.lemmatize(word) for word in lauseen_sanat]
    bag = [0] * len(words)
    for w in lauseen_sanat:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def ennakoi_luokka(lause, model):
    bow = kasa_sanoja(lause, words, show_details=False)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def saa_vastaus(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def botin_vastaus(msg):
    ints = ennakoi_luokka(msg, model)
    res = saa_vastaus(ints, intents)
    return res

def puheTekstiksi():
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language='fi-FI')
        return text
    except:
        return 'Botti ei tainnut ymmärtää mitä sanoin'

def tekstia():

    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "Sinä: " + msg + '\n\n')
        ChatBox.config(foreground="#442265", font=("Verdana", 12))

        res = botin_vastaus(msg)
        ChatBox.insert(END, "Botti: " + res + '\n\n')

        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


def puhetta():
    msg = puheTekstiksi()

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "Sinä: " + msg + '\n\n')
        ChatBox.config(foreground="#442265", font=("Verdana", 12))

        res = botin_vastaus(msg)
        ChatBox.insert(END, "Botti: " + res + '\n\n')

        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)

#GUI

root = Tk()
EntryBox = Text(root, bd=0, bg="white", width="40", height="10", font="Arial")

root.title("Lopputehtävä Chatti-botti")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial", )
ChatBox.config(state=DISABLED)

scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="hand1")
ChatBox['yscrollcommand'] = scrollbar.set

PuheButton = Button(root, font=("Verdana", 12, 'bold'), text="Puhu", width="12", height=5,
                   bd=0, bg="#32de97", activebackground="#3c9d9b", fg='black', command=puhetta)

LahetaButton = Button(root, font=("Verdana", 12, 'bold'), text="Lähetä", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='black', command=tekstia)

EntryBox = Text(root, bd=0, bg="white", width="40", height="10", font="Arial")
EntryBox.bind("<Return>", (lambda event: tekstia()))

scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
PuheButton.place(x=6, y=401, height=40)
LahetaButton.place(x=6, y=450, height=40)
root.mainloop()