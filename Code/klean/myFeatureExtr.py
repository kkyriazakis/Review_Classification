import gzip
import simplejson
import nltk
from pattern.en import singularize

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


# ---- MAIN ----
pos_tags_file = open("pos_tags_file.txt", "w")  # a+ -> append
nouns_file = open("nouns_file.txt", "w")        # a+ -> append

index = list()
itemCounter = 0
item = "\"review/text\": \""

for e in parse("Shoes.txt.gz"):
    ent = simplejson.dumps(e)
    start = ent.find(item)+len(item)
    end = ent.find("\"}")
    if itemCounter == 10:
        break
    sentence = ent[start:end]
    if start != -1:
        itemCounter += 1
        tokens = nltk.word_tokenize(sentence)   # split sentence into words
        tagged = nltk.pos_tag(tokens)   # tag tokens
        # print(tagged)

        is_noun = lambda pos: pos[:2] == 'NN'
        nouns = [word for (word, pos) in tagged if is_noun(pos)]    # find nouns in sentence
        # print(nouns)

        for i in tagged:
            pos_tags_file.write(repr(i) + "\n")
        for i in nouns:
            nouns_file.write(i.lower() + "\n")

pos_tags_file.close()
nouns_file.close()

Dictionary = open("dictionary.txt", "w")
nouns_file = open("nouns_file.txt", "r")
lineCount = 0
for line in open("nouns_file.txt").readlines(): lineCount += 1

d = dict()
for i in range(lineCount):
    content = nouns_file.readline().rstrip()    # read new line without "\n"
    content = singularize(content)  # plural to singular

    if content not in d:
        d[content] = 1
    else:
        d[content] = d[content] + 1

d = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
print(d)

for i in d:
    Dictionary.write(repr(i) + "\n")

Dictionary.close()

