from __future__ import division
import nltk
import numpy
import glob
from nltk.collocations import *

# Empty list to hold all of the terms across documents, and another
# to hold text documents.
all_terms = []
texts = []
lentexts = []

# The directory to be parsed
path = "corpus/*.split"
directories = glob.glob(path)
print "Tokenizing documents in the directories:", directories, "\n"

# Iterate through the directory and build the collection of texts and a list
# of terms for NLTK.
for directory in directories:
    print "-----------------------------------------------------------"
    print "Now tokenizing the directory:", directory
    print "-----------------------------------------------------------"
    files = glob.glob(directory + '/*')
    for infile in files:
        print "Now tokenizing the file:", infile
        if infile.startswith('.'): #Mac directories ALWAYS have a .DS_Store file.
            continue               #This ignores it and other hidden files.
        f = open(infile);
        raw = f.read()
        f.close()
        tokens = nltk.word_tokenize(raw)
        tritokens = nltk.trigrams(tokens)
        text = nltk.Text(tritokens)
        texts.append(text)
print "-----------------------------------------------------------"

# First we need to get the frequent collocations in the entire corpus
print "Computing the frequent bigrams of the corpus"
trigram_measures = nltk.collocations.TrigramAssocMeasures()
corpus = open('corpus.txt').read()
finder = TrigramCollocationFinder.from_words(corpus.split())
scoredTrigram = finder.nbest(trigram_measures.raw_freq, 2000)

print "Using", len(scoredTrigram), "frequent items"

# Function to create a BOW for one document.  For each of
# our unique words, we have a feature which is the count for that word
def BOWf(document):
    word_counts = []
    for trigram in scoredTrigram:
        word_counts.append(document.count(trigram))
    return word_counts

### And here we actually call the function and create our list of vectors.
print "Creating Numpy array vectors"
vectors = [numpy.array(BOWf(f)) for f in texts]
