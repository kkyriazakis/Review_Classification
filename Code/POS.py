import gzip, simplejson, re, pickle,nltk

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


index=list()
itemCounter =0
inserCounter =0
item = "\"review/text\": \""
for e in parse("Cell_Phones_&_Accessories.txt.gz"):
    ent= simplejson.dumps(e)
    start=ent.find(item)+len(item)
    end = ent.find("\"}")
    if (itemCounter==1):
        break
    sentence = ent[start:end]
    if start != -1:
        tokens = nltk.word_tokenize(sentence)

        tagged = nltk.pos_tag(tokens)
        print(tagged)
        itemCounter+=1
        inserCounter+=len(tokens)



