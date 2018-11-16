from __future__ import print_function
from __future__ import division

from nltk.corpus import brown, conll2000, floresta,treebank, dependency_treebank, alpino
from  nltk import FreqDist
from nltk import bigrams

#**
# get part of speech and the number of each tag
sents = brown.tagged_sents(tagset='universal')
sentences = sents[:10000]
print('=== How many sentences in the corpus: ', len(sents),' ===' )
tags = []

for sen in sentences:
    tags.append("<s>") #add the start-of-sentence
    tags += [t for (_,t) in sen]
    tags.append("</s>") #add the end-of-sentence

fdis1 = FreqDist(tags)
pos = dict(fdis1.most_common())
print ('=== How many different types of POS in the corpus: ', len(pos)-2,' ===')

totalUni = 0
for t in pos:
    if t != "<s>" and t != "</s>":
        totalUni += pos[t] #calculate the number of POS

#**
# make transitions
tags1 = list(bigrams(tags))
fdis1_ = FreqDist(tags1)
transitions = dict(fdis1_.most_common())

del transitions[('</s>','<s>')]

#**
## transfer transtions to transititon probabilities
tranProba = {} # store transition probabilities
for (t1,t2) in transitions:
    #tranProba[t1,t2] = ((transitions[t1,t2]) + 1) / (pos[t1]+len(pos)) # Laplace smoothing
    tranProba[t1,t2] = (transitions[t1,t2]) / pos[t1] # Stupic backoff

#**
# make emissions
words1 = [] # store all words from the old training sentences
words2 = {} # store all words and the number of the occurence of each word
for sen in sentences:
    words1 += [w for (w,_) in sen]
fdis2 = FreqDist(words1)
words2 = dict(fdis2.most_common())

occur_once = {}
for w in words2:
    if words2[w] == 1:
        occur_once[w] = 1 # get words that each of them occurs once.

newWords = [] # the new training sentences with tags
for sen in sentences:
    for w in sen:
        listW = list(w)
        try:
            occur_once[listW[0]]
            listW[0] = 'unknown' #replace the words that occur once to the word "unknown"
        except: listW[0] = listW[0]
        newWords.append(tuple(listW))


pairs = []
for sen in newWords:
    pairs.append(sen)

fdis3 = FreqDist(pairs)
emissions = dict(fdis3.most_common())

#**
# transfer emissions to emission probabilities
emiProba = {} # store emission probabilities
for (word,tag) in emissions:
        emiProba[word,tag] = (emissions[word,tag]) / (pos[tag])

#**
# make a dictionary storinng words and the corresponding tags
dictionary = {}

for (word,tag) in newWords:
    if word not in dictionary:
        dictionary[word] = [tag]
    else:
        if tag not in dictionary[word]:
            dictionary[word].append(tag)

#**
# the HMM model can tag words in a sentence, and use Viterbi algorithm
def tagger(test_sentence):
# use algorithm to find proper "tag path"
    temp_probability = 0 # store probabilities of different tags of a word
    probability = 0 # store the maximum probability
    predicted_tags = {}
    preTags = []
    for i,w in enumerate(test_sentence):
        if i == 0:
            if w not in dictionary:
                w = 'unknown'
            predicted_tags[i] = dictionary[w][0] # use the first possible POS of w as initial predicted_tags

            for tag in dictionary[w]:
                em = 0.0
                em = emiProba[w,tag] # emission probability of w, i.e. p(w|tag)

                tr = 0.0
                try:
                    tr = tranProba['<s>',tag]
                except KeyError:
                    #tr = 1 / (pos['<s>'] + len(pos)) # Laplace smoothing
                    tr = (pos[tag]/totalUni) * 0.4 # Stupic backoff

                temp_probability = tr*em
                if temp_probability >= probability:
                    probability = temp_probability # update the probability when the new probability is bigger than the old probability
                    predicted_tags[i] = tag # updat the tag

        else:
            if w not in dictionary:
                w = 'unknown'
            predicted_tags[i] = dictionary[w][0]

            for tag in dictionary[w]:
                em = 0.0
                em = emiProba[w,tag]

                tr = 0.0
                try:
                    tr = tranProba[predicted_tags[i-1],tag] # p(tagi|tagi-1)
                except KeyError:
                    #tr = 1 / (pos[predicted_tags[i-1]]+len(pos))
                    tr = (pos[tag]/totalUni) * 0.4
                temp_probability = tr*em
                if temp_probability >= probability:
                    probability = temp_probability
                    predicted_tags[i] = tag

    for k in predicted_tags:
        preTags.append(predicted_tags[k])
    return preTags

#**
# calculate the accuracy rate
def CompareAccuracy(real_tags,tag_list):
    errors = 0
    accuracy = 0.0

    for i,t in enumerate(real_tags): # compare if ta_list is identical to real_tags
        #print(t, ' - ', tag_list[i])
        if t == tag_list[i]:
            errors +=1 #sum of errors
    print ('the number of tags in the test sentences: ', len(real_tags))
    accuracy = errors/len(real_tags)
    return accuracy

#**
# calculate the accuracy rate of POS tagging
def AccuracyOfPOS(real_tags,tag_list):
    compare_tags = {}

    # utilise the tags in pos to fill up the keys in compare_tags
    for t in pos:
        compare_tags[t] = [0,0] # keys in compare_tags include all universal tags from brown
    del compare_tags['<s>']
    del compare_tags['</s>']

    for i,t in enumerate(real_tags):
        compare_tags[t][0] +=1
        if t == tag_list[i]:
            compare_tags[t][1] +=1
    for t in compare_tags:
        try:
            compare_tags[t] = (compare_tags[t][1]/compare_tags[t][0])
        except:
            compare_tags[t] = 'not_exist'

    return compare_tags

#**
# test the xth-yth sentences from brown corpora
sentence_list = sents[22469:24007]
real_tags = []
tag_list = []

for sen in sentence_list:
    sentence = []
    real_tags += [t for (_,t) in sen] # get the list of the real tags of the words in sentence_list
    #print ("888888888", real_tags)
    sentence += [w for (w,_) in sen]
    tag_list+=(tagger(sentence)) # get the list of predicted_tags of the words in sentence_list
print ('Accuracy: ', CompareAccuracy(real_tags, tag_list))
print ("")
print ('Accuracy of each tags: ', AccuracyOfPOS(real_tags, tag_list))
