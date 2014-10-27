
# coding: utf-8

# In[1]:

import re
import random
from os import path

import nltk


# In[ ]:

path_train = path.join(path.curdir, "train.txt")
path_final_testing = path.join(path.curdir, "test.csv")


# In[2]:

def file_to_sets(fname, ignore_header=True):
    """
    Takes a file where each line is in the format "category,text" and turns it into a list of tuples
    in format "(category, text)"
    """
    
    f = open(fname, 'r')
    
    if ignore_header:
        # This skips the first line of the file
        next(f)
    
    out = []
    for line in f:
        # iterate over lines, use simple regex to separate the category from text
        out.append(re.match(r"(\d+),(.+$)", line).groups())
    
    f.close()
    return out


# In[ ]:

sample_sets = file_to_sets(path_train, ignore_header=False)
final_sets = file_to_sets(path_final_testing, ignore_header=True)


# In[3]:

def get_sets(samples, test_fraction=3):
    """
    takes a set of samples, shuffles them, then returns two lists, train_sets and test_sets. 
    The size of test_sets is len(samples)/test_fraction, train_sets is the remainder. 
    """
    
    # don't shuffle the sample list as that will affect the list passed in
    l = samples[:]
    random.shuffle(l)
    
    test_size = int(len(l)/test_fraction)
    test_sets = l[0:test_size]
    train_sets = l[test_size:]
    
    return train_sets, test_sets


# In[4]:

def get_folds(samples, folds=3):
    """
    Returns of list of folds from samples
    """
    
    # don't shuffle the sample list as that will affect the list passed in
    l = samples[:]
    random.shuffle(l)
    out = []
    chunk_size = int(len(samples)/folds)
    sections = range(0, len(samples)+1, chunk_size)
    sections[-1]= None
    
    for i in range(0,len(sections)-1):
        out.append(l[sections[i]:sections[i+1]])
    
    return out


# In[9]:

class FeatureExtractor(object):
    """A class to make it easy to combine and shuffle around feature extractors"""
    
    def __init__(self, extractors=None):
        """
        Takes a list of extractors to use in extracting features. 
        Extractors should take a piece of text and return a dictionary where the key is
        the desired key and the value is the feature value. 
        """
        if extractors is None:
            extractors = []
            
        if type(extractors) is not list:
            extractors = [extractors]
            
        self.extractors = extractors
        
    def __call__(self, text):
        features = {}
        for e in self.extractors:
            f = e(text)
            for k, v in f.iteritems():
                features[k]=v
        
        return features
    
    def add_extractor(self,extractor):
        self.extractors.append(extractor)


# In[ ]:

stopwords = nltk.corpus.stopwords.words('english')

def get_terms(t):
    tokens = nltk.word_tokenize(t)
    return [w for w in tokens if w not in stopwords]


# In[6]:

def create_training_sets (feature_function, items):
    # Create the features sets.  Call the function that was passed in.
    # For names, key is the name, and value is the gender
    featuresets = [(feature_function(key), value) for (key, value) in items]
    
    # Divided training and testing in half.  Could divide in other proportions instead.
    halfsize = int(float(len(featuresets)) / 2.0)
    train_set, test_set = featuresets[halfsize:], featuresets[:halfsize]
    return train_set, test_set


# In[26]:

def make_classifier(feature_extractor, train, classifier=nltk.classify.NaiveBayesClassifier):
    """
    creates a classifier based on the feature_extractor, trains it with train and
    tests it with test
    """
    train_features = [(feature_extractor(text), category) for (category, text) in train]
    
    cl = classifier.train(train_features)
    return cl


# In[12]:

def fold_test_extractor(feature_extractor, samples, folds=3):
    """
    Tests a feature extractor with a set of sample tuples in format (category, text)

    Params:
    feature_extractor -- The feature extractor function to use
    samples -- the samples to test with
    folds -- the number of folds to use in testing
    """
    
    features = [(feature_extractor(text), category) for category, text in samples]
    folds = get_folds(features, folds)
    
    for i in range(0, len(folds)):
        train = [f for idx, s in enumerate(folds) for f in s if idx !=i]
        test = folds[i]
        
        cl = nltk.NaiveBayesClassifier.train(train)
        print "test {} - {:.3%}".format(i, nltk.classify.accuracy(cl, test))


# In[28]:

def make_submission(classifier, samples, writeto=None):
    out = []
    for s in samples:
        out.append((s[1], classifier.classify(s[0])))
    
    if writeto:
        out_file = open(writeto, 'w')
        out_file.write("Id,Category\n")
        for n, c in out:
            out_file.write("{},{}\n".format(n,c))
    
    return out


# In[27]:

def make_feature(extractor, samples):
    return [(extractor(text), category) for category, text in samples]

