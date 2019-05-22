import gzip
import simplejson
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import gzip
import simplejson
import json
import _pickle as pickle
import nltk


# Source: http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


def parse(filename):
    f = gzip.open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(b':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos+2:]
        entry[eName] = rest
    yield entry

def remove_unwanted_char(txt):
    txt = re.sub(r'[\"\'<>+=_@#%$&*}{~`/|()^.,]', '', txt)
    txt = txt.replace('-', '')   # removes -
    txt = txt.replace('\\', '')  # removes \
    txt = " ".join(txt.split())  # removes multiple spaces
    return txt

def clean_text(sent, lab, remove_stopwords=True):
    sent = sent.lower()  # lowercase sentence
    lab = lab.lower()  # lowercase label

    # replace contractions
    sent = sent.split()
    lab = lab.split()
    temp = []
    for w in sent:
        if w in contractions:
            temp.append(contractions[w])
        else:
            temp.append(w)
    sent = " ".join(temp)
    temp = []
    for w in lab:
        if w in contractions:
            temp.append(contractions[w])
        else:
            temp.append(w)
    lab = " ".join(temp)

    # remove unwanted characters contractions
    sent = remove_unwanted_char(sent)
    lab = remove_unwanted_char(lab)

    # Remove stop words from sentence only
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        sent = sent.split()
        sent = [w for w in sent if not w in stops]
        sent = " ".join(sent)

    return sent, lab

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1



itemCounter = 0
# ---- MAIN ----
file = "Watches.txt.gz"
label_start = "\"review/summary\": \""
review_start = "\"review/text\": \""
sentences = []
labels = []
threshold=2

for e in parse(file):
    ent = simplejson.dumps(e)

    l_start = ent.find(label_start) + len(label_start)
    l_end = ent.find(review_start) - 3
    label = ent[l_start:l_end]  # extract label

    s_start = ent.find(review_start)+len(review_start)
    s_end = ent.find("\"}")
    sentence = ent[s_start:s_end]  # extract review

    if s_start != -1 and l_start != -1:

        sentence, label = clean_text(sentence, label, remove_stopwords=True)
        # sentence einai olo to review ka8arismeno
        # label einai to summary ka8arismeno
        sentences.append(sentence)
        labels.append(label)
        itemCounter += 1

word_dict={}

count_words(word_dict,sentences)
count_words(word_dict, labels)

print("Size of Vocabulary:", len(word_dict))
with open('Vocabulary','wb') as fp:
    pickle.dump(word_dict,fp)


# embeddings_index = {}
# with open('numberbatch.txt', encoding='utf-8') as f:
#     for line in f:
#         values = line.split(' ')
#         word = values[0]
#         embedding = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = embedding
#
# print('Word embeddings:', len(embeddings_index))
#
#
# with open('embIndex_pickle','wb') as fp:
#     pickle.dump(embeddings_index,fp)

with open('embIndex_pickle','rb') as fp:
    embeddings_index=pickle.load(fp)

print(len(embeddings_index))

vocab_to_int = {}

value = 0
for word, count in word_dict.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_dict),4)*100

print("Total number of unique words:", len(word_dict))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))


embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))
with open('emb_matrix','wb') as fp:
    pickle.dump(word_embedding_matrix,fp)
