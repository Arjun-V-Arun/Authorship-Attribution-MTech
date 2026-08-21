from __future__ import division
import nltk
import numpy
import glob

# Empty list to hold all of the terms across documents, and another
# to hold text documents.
all_terms = []
texts = []

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
        tokens = nltk.word_tokenize(raw) # add individual tokens to list of tokens
        all_terms = all_terms + tokens  # add individual tokens to list of tokens
        text = nltk.Text(tokens) # turn the list of tokens into an nltk.Text object
        texts.append(text) # and add to the list of nltk.Text objects
print "-----------------------------------------------------------"

#get rid of duplicate tokens
all_terms = list(set(all_terms))

# Get a list of the most frequent words in the corpus
frequent_terms_doc = open('frequent_words.txt').read()
frequent_terms = nltk.word_tokenize(frequent_terms_doc)
frequent_terms = list(set(frequent_terms))

print "Found a total of", len(all_terms), "unique terms and using", len(frequent_terms), "frequent items"
print "Prepared", len(texts), "documents..."
print "They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]"

# Function to create a BOW for one document.  For each of
# our unique words, we have a feature which is the count for that word
def BOW(document):
    word_counts = []
    for word in all_terms:
        word_counts.append(document.count(word))
    return word_counts

# Function to create a BOW for one document.  For each of
# our unique words, we have a feature which is the count for that word
def BOWf(document):
    word_counts = []
    for word in frequent_terms:
        word_counts.append(document.count(word))
    return word_counts

### And here we actually call the function and create our list of vectors.
vectors = [numpy.array(BOWf(f)) for f in texts]
print "Vectors created."
