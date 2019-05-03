import gzip, simplejson, re, pickle

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

def format(filename):
    index=list()
    itemCounter =0
    inserCounter =0
    item = "\"review/text\": \""
    for e in parse(filename):
        ent= simplejson.dumps(e)
        start=ent.find(item)+len(item)
        end = ent.find("\"}")
        if (itemCounter==1):
            break
        if start != -1:
            reFormated=re.sub(r'[\"<>+=_@#%$&*}{~`.!,;?/|()^-]', " ",ent[start:end]).split()
            itemCounter+=1
            inserCounter+=len(reFormated)
        print(reFormated)
        for x in reFormated:
            index.append(x)
    return list(set(index))


uniIndex = format("Shoes.txt.gz")
print(uniIndex)

int=list()
for i in range(len(uniIndex)):
    int.append(i)

Dictionary=list()

Dictionary=[(uniIndex[i],int[i]) for i in range(0,len(uniIndex))]


print(len(Dictionary))

with open('Index','wb') as fp:
    pickle.dump(Dictionary,fp)


with open('Index','rb') as fp:
    recovered=pickle.load(fp)

print(len(recovered))


