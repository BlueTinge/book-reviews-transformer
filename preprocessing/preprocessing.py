import json
import gzip
from nltk.tokenize import word_tokenize
import re

# Open zipped .json file and use as
# strict json
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))
    
# Open unzipped .json file
def parse2(path):
  g = open(path, 'r')
  for l in g:
    yield eval(l)

# Open unzipped .json file and use as
# strict json
def parse3(path):
  g = open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))


# Script to trim review data to reasonable size
'''f = open("outputnew.json", 'w')
i = 0
for l in parse3("output.json"):
  if i == 500000:
    break
  f.write(l + '\n')
  i += 1'''

# Script to count total number of reviews in
# our trimmed dataset
'''total = 0
for review in parse2("outputnewCopy.json"):
  total+=1'''

# Script to remove junk punctuation from
# our review dataset
'''i = 0
f = open("finalremoving.json", 'w')
newreviews = []
for review in parse2("outputnewCopy.json"):
  reviewtext = review['reviewText']
  splittext = reviewtext.split(" ")
  newtext = []
  for word in splittext:
    newword = str(re.sub(r'\,?(\.\.\.)?(\\\")?\-?\(?\)?\[?\]?\:?\;?\{?\}?\/?\\?\|?(\&quot\;)?(\&\#34\;)?\*?\"?(\.\.)?', r'', word))
    newtext.append(newword)
  space = " "
  space = space.join(newtext)
  review['reviewText'] = space
  dumper = json.dumps(review)
  f.write(dumper)
  f.write('\n')
  i+=1
  print((i/total)*100)
f.close()'''


# Script to make a tokenized csv file of data for testing
# and training, and a pure tokenized text file for 
# training word embeddings
'''i = 0
f = open("removedreviews.csv", 'w')
f2 = open("pureremovedreviews.txt","w")
for review in parse2("finalremoving.json"):
  text = review['reviewText']
  tokenizedtext = word_tokenize(text)
  f.write("\"")
  for word in tokenizedtext:
    f.write(word)
    f2.write(word)
    f.write(" ")
    f2.write(" ")
  f.write("\","+str(int(review['overall']-1.0)))
  f.write('\n')
  f2.write('\n')
  i+=1
  print((i/total)* 100)
f.close()
f2.close()'''

# Script to count total number of reviews in dataset
'''f = open("removedreviews.csv", 'r')
i = 0
for line in f:
  i+=1
f.close()'''

# Script to split dataset in half for testing and training
'''f = open('removedreviews.csv', 'r')
fifty = i//2
j = 0
train = open('training.csv', 'w')
test = open('testing.csv', 'w')
for line in f:
    if j < fifty:
      train.write(line)
    elif j >= fifty:
      test.write(line)
    j+=1
train.close()
test.close()'''

# Scripts to calculate and print out review distributions
# for testing and training datasets individually
'''train = open('training.csv', 'r')
test = open('testing.csv', 'r')
totaltrain = 0
totaltest = 0
traindist = [0]*5
testdist = [0]*5
for line in train:
  combo = line.split(',')
  if(len(combo) > 2):
    print("ERROR")
    continue
  score = int(combo[1])
  traindist[score] += 1
  totaltrain+=1
print(traindist)
for i in range(len(traindist)):
  traindist[i] = (traindist[i] / totaltrain) * 100
print(traindist)

for line in test:
  combo = line.split(',')
  if(len(combo) > 2):
    print("ERROR")
    continue
  score = int(combo[1])
  testdist[score] += 1
  totaltest+=1
print(testdist)
for i in range(len(testdist)):
  testdist[i] = (testdist[i] / totaltest) * 100
print(testdist)'''
  
  
# Listed below are the invalid punctuation characters
# removed from our review data:

# , ... \" - ( ) [ ] : ; { } / \ | &quot; &#34; * "
