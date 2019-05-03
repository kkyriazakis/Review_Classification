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


pos_tags_file = open("pos_tags_file.txt", "w")

index=list()
itemCounter =0
inserCounter =0
item = "\"review/text\": \""
for e in parse("Shoes.txt.gz"):
    ent= simplejson.dumps(e)
    start=ent.find(item)+len(item)
    end = ent.find("\"}")
    if (itemCounter==1):
        break
    sentence = ent[start:end]
    if start != -1:
        tokens = nltk.word_tokenize(sentence)
        print(tokens)
        tagged = nltk.pos_tag(tokens)
        print(tagged)
        itemCounter+=1
        inserCounter+=len(tokens)
        for i in tagged:
            pos_tags_file.write(repr(i) + "\n")


pos_tags_file.close()


