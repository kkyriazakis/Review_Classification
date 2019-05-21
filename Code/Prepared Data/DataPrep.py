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


itemCounter = 0
# ---- MAIN ----
file = "Watches.txt.gz"
label_start = "\"review/summary\": \""
review_start = "\"review/text\": \""

for e in parse(file):
    ent = simplejson.dumps(e)

    l_start = ent.find(label_start) + len(label_start)
    l_end = ent.find(review_start) - 3
    label = ent[l_start:l_end]  # extract label

    s_start = ent.find(review_start)+len(review_start)
    s_end = ent.find("\"}")
    sentence = ent[s_start:s_end]  # extract review

    if itemCounter == 1:
        break
    if s_start != -1 and l_start != -1:
        itemCounter += 1
