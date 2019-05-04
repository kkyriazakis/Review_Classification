from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import glob
from nltk import pos_tag, stem
from math import log10
from future.utils import iteritems
threshold = 5000

tokenizer = RegexpTokenizer(r'[\w\']+')
get_token = tokenizer.tokenize
snowball = stem.RegexpStemmer('ies$|s$')
swlist = stopwords.words('english')
noun_file_pointer = open("tokenized_noun_file.txt", "w");

noun_postags, tf = [], {}
curr_line = 0
tot=27061597
percent = 0

pos_tags_file = open("pos_tags_file.txt", "r")
line = pos_tags_file.readline()
while line:
    i = eval(line)
    if i[1].find("NN") != -1:
        noun_postags.append(i[0].strip(".,-?").lower())
    line = pos_tags_file.readline()
    curr_line += 1
pos_tags_file.close()

print("Counting nouns...")

for i in noun_postags:
    if snowball.stem(i) in tf:
        tf[snowball.stem(i)] += 1
    else:
        tf[snowball.stem(i)] = 1

noun_postags = []

print("Sorting nouns")


for token, count in tf.items():
    if count < threshold or (token in swlist) or len(token) < 4:
        continue
    noun_postags.append((count, token))
noun_postags.sort()
noun_postags.reverse()


print("Writing Features")

for count, token in noun_postags:
    noun_file_pointer.write(token+ ":  "+ repr(count) + "\n")

noun_file_pointer.close()
print("Process over")