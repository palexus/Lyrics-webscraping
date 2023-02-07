""" Program to download lyrics
"""

import argparse
from bs4 import BeautifulSoup
from requests.exceptions import ConnectionError
import os
import numpy as np
import requests
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import MultinomialNB
import nltk  
from nltk.tokenize import TreebankWordTokenizer # very good tokenizer for english, considers sentence structure
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer   


PREURL = "https://www.lyrics.com/"
STOPWORDS = stopwords.words('english')


def get_artist_url(artist_name):
    art_link_list = []
    try:
        search_url = PREURL + "lyrics/" + artist_name
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all(class_="name")
        print(f'There are {len(links)} artist links')
        for i in range(len(links)):
            print(f'The name of the artist {i + 1} is {links[i]["title"]}')
            artist_url = PREURL + links[i]["href"]
            art_link_list.append(artist_url)
    except ConnectionError as e:
        print(e)
    return art_link_list


def get_song_urls(artist_url):
    """
        Takes the artist page and return a list of songtitles and
        a list with the corresponding songurls.
    """
    response = requests.get(artist_url)
    soup = BeautifulSoup(response.text, "html.parser")
    names, links = [], []
    suppies = soup.find_all(class_="tal qx")
    for s in suppies:
        names.append(s.get_text())
        links.append(s.find(href=True)["href"])
    return names, links


def clean_from_duplicates(names, links):
    names = np.array(names)
    links = np.array(links)
    unique_names = set()
    indices = []
    for i, song in enumerate(names):
        if song not in unique_names:
            indices.append(i)
            unique_names.add(song)
    return names[indices], links[indices]


def write_to_disk(artist, names, links):
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, "lyrics", artist)
    if not os.path.join(parent_dir, "lyrics"):
        print("Wieso macht der nichts?")
        os.makedirs(os.path.join(parent_dir, "lyrics"))
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"Directory {path} is created")
    expr = '<pre[\S\s]+pre>'
    for i, l in enumerate(links):
        lyric_url = PREURL + l
        response = requests.get(lyric_url)
        song_title = names[i]
        try:
            prebox = re.findall(expr, response.text)[0]
            soup = BeautifulSoup(prebox,features="html.parser")
            text = soup.get_text()
            with open(os.path.join(path, f'{song_title}.txt'), 'w') as f:
                f.write(text)
        except (IndexError, ConnectionError, FileNotFoundError) as e:
            print(e)
            print("The lyrics of the song "+names[i]+" are missing")
        else:
            print(i, song_title, response.status_code)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def clean_text(content):
    content = content.replace("\n", " ")
    content = content.lower()
    content = re.sub('[^A-Za-z\s]+', '', content)
    content = re.sub(' +', ' ', content)
    return content


def clean_titles(corpus, songs, threshold=0.33):
    zipped = list(zip(songs, corpus))
    zipped.sort()
    songs, corpus = list(zip(*zipped))
    clean_songs = [songs[0]]
    clean_corpus = [corpus[0]]
    for i in range(len(songs)-1):
        if similar(songs[i], songs[i+1])<threshold:
            clean_songs.append(songs[i+1])
            clean_corpus.append(corpus[i+1])
    return clean_corpus, clean_songs


def get_corpus_and_names(path):
    corpus, names = [], []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if not os.path.isfile(filepath):
            continue
        with open(filepath, 'r') as f:
            names.append(filename.rstrip(".txt"))
            content = f.read()
            content = clean_text(content)
            corpus.append(content)
    return corpus, names


def concatenate(artists, corpera):
    labels = []
    corpus = np.concatenate(corpera)
    for i in range(len(corpera)):
        labels += len(corpera[i]) * [artists[i]]
    return corpus, labels


def wordpredicter(word):
    return m.predict(vectorizer.transform([word]))


def print_score(X, y, models):
    if type(models)!=list:
        models = [models]
    for m in models:
        y_pred = m.predict(X)
        print('\033[1m' + "The Score for the model: " + '\033[0m', type(m).__name__ , "\n")
        print("accuracy: ", accuracy_score(y, y_pred))
        print(classification_report(y, y_pred, target_names=np.unique(y), zero_division=1))

        cm = confusion_matrix(y, y_pred, normalize="pred")
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()
        plt.close()




### HERE STARTS THE PROGRAM ###

parser = argparse.ArgumentParser(description = __doc__, epilog = 'some final remarks')

parser.add_argument("artist", help="Please enter the name of at least two artists.", nargs="+")
parser.add_argument("--like", help="How do you like the artist at all?")


args = parser.parse_args()
artist = args.artist

parent_dir = os.getcwd()


corpera, namings = [], []

for art in artist:
    path = os.path.join(parent_dir, "lyrics", str(art))
    if os.path.isdir(path):
        dl = input(f"\n The folder of the artist {art} already exists. Do you still want to download? [y/n]: ")
        if dl=="y":
            art_urls = get_artist_url(art)
            k = int(input("How many do you want to inspect? "))
            for i in range(k):
                names, _ = get_song_urls(art_urls[k])
                print(f'The link number {i} has {len(names)} songs')
            m = int(input("\n Which link do you choose?"))
            names, links = get_song_urls(art_urls[m])
            print(f'\n {len(links)} song urls of the artist {art} found! (duplicates possible)')
            names, links = clean_from_duplicates(names, links)
            print(f'{len(links)} song urls of the artist {art} found! (cleaned)\n')
            sure = input(f"Sure you want to save them to {path}?")
            if sure=="y":
                print("Write lyrics to disk!")
                write_to_disk(art, names, links)
            else:
                print("ok, then not!")
            corpus, names = get_corpus_and_names(path)
            corpus, names = clean_titles(corpus, names)
            corpera.append(corpus)
            namings.append(names)
        else:
            corpus, names = get_corpus_and_names(path)
            corpus, names = clean_titles(corpus, names)
            corpera.append(corpus)
            namings.append(names)
    else:
        dl = input(f"\n The folder of the artist {art} does not exists. Do you want to create an folder and download some files? [y/n]: ")
        if dl in ["y", " y", "yes", " yes"]:
            art_urls = get_artist_url(art)
            k = int(input("How many do you want to inspect? "))
            for i in range(k):
                names, _ = get_song_urls(art_urls[i])
                print(f'The link number {i} has {len(names)} songs')
            m = int(input("\n Which link do you choose?"))
            names, links = get_song_urls(art_urls[m])
            print(f'\n {len(links)} song urls of the artist {art} found! (duplicates possible)')
            names, links = clean_from_duplicates(names, links)
            print(f'{len(links)} song urls of the artist {art} found! (cleaned)\n')
            sure = input(f"Sure you want to save them to {path}?")
            if sure in ["y", " y", "yes", " yes"]:
                print("Write lyrics to disk!")
                write_to_disk(art, names, links)
            else:
                print("ok, then not!")
            corpus, names = get_corpus_and_names(path)
            corpus, names = clean_titles(corpus, names)
            corpera.append(corpus)
            namings.append(names)
        else:
            print("\n This is not how this game works!")
            quit()


#def concatenate(art_1, art_2, corpus_1, corpus_2):
 #   corpus = corpus_1 + corpus_2
  #  labels = len(corpus_1) * [art_1] + len(corpus_2) * [art_2]
  #  return corpus, labels



corpus, labels = concatenate(args.artist, corpera)

 # fit bag of words model on our corpus



X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=10)

vectorizer = TfidfVectorizer(stop_words=STOPWORDS) # instanciation
vectors_train = vectorizer.fit_transform(X_train) 
vectors_test = vectorizer.transform(X_test)

m = MultinomialNB()
m.fit(vectors_train, y_train) 


print_score(vectors_train, y_train, m)
print_score(vectors_test, y_test, m)


print("#################################")
print("###      OK LETS PLAY!!!      ###")
print("#################################")

while True:
    word = input("Type in some word! ")
    pred = m.predict(vectorizer.transform([word]))
    print(f'It is probably from this artist: ')
    print()
    print(pred[0])
    print()