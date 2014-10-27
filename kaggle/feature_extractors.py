
# coding: utf-8

# In[2]:

import re
import random
from os import path

from testing_util import get_terms
import nltk


# In[ ]:

def category_term_scorer(sample_list):
    """
    takes a list of tuples in format (category,text) and creates scores for each term
    for its relevance to each category
    """
    categories = {}
    
    for c,s in sample_list:
        if c not in categories:
            categories[c]=[]
        for w in get_terms(s):
            categories[c].append(w)
    
    fd_all = nltk.FreqDist([w for wl in categories.values() for w in wl])
    
    fd_categories = {c:nltk.FreqDist(v) for c,v in categories.iteritems()}
    
    term_scores = {}
    for term in fd_all.iterkeys():
        d = {}
        for c,fd in fd_categories.iteritems():
            d[c]= 1 if fd.freq(term) > fd_all.freq(term) else 0
        term_scores[term]=d
    
    return term_scores


# In[ ]:

class TermScoreClassiffier(nltk.classify.ClassifierI):
    """
    Tries to classify text using scored terms. 
    """
    
    def __init__(self, samples=None, scorer=category_term_scorer, terms=None, key="TermScore"):
        """
        Params:
        
        samples -- a list of samples where each entry is a tuple in format (category,text)
                this argument only works if scorer is also passed. 
                
        scorer -- a function that takes the list of samples and scores them. Must return a dictionary
                in the same format as terms
        
        terms -- a dictionary of terms where keys are the terms and values are dictionaries 
        with the score for each category. ie: {"term": {"c1":0, "c2":10}
        
        key -- The key to used in the returned dictionary. 
        """
        self.key = key
        
        if samples and scorer:
            terms = scorer(samples)
        
        if not terms:
            raise ValueError("You must either pass a list of samples or a list of terms")
        
        self.terms = terms
    
    def __call__(self, text):
        """
        Picks a category for text using the term list
        """
        
        tokens = nltk.word_tokenize(text)
        scores = {}
        for w in tokens:
            if w in self.terms:
                for c,s in self.terms[w].iteritems():
                    if c in scores:
                        scores[c] += s
                    else:
                        scores[c] = s
        
        totals = scores.items()
        totals.sort(key= lambda s:s[1], reverse=True)
        
        return {self.key: totals[0][0]}

