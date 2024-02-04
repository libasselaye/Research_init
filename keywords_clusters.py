# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:35:58 2021

@author: lamot
"""

import os
os.chdir("D:/M1-S2/Init Recherche/projet_init_recherche/dataset")
import nltk
import pandas as pd
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


data = pd.read_csv("groupes.csv")

cluster_kwlist = pd.DataFrame(data.groupby('groupe')['keywords'].apply(' '.join))

from collections import Counter        
        
split_it = cluster_kwlist['keywords'][1].split()
split_it

Counter = Counter(split_it)

most_occur = Counter.most_common(10)
mots_mo = ' '.join([x[0] for x in most_occur])

#Bi-grammes function 

#text étant la liste déja nettoyer avec nettoyer_texte

def ngrames(sentence,n):
    ngramms = []
    ngrm = nltk.ngrams(sentence.split(), n)
    for grm in ngrm:
        ngramms.append(list(grm))
    return ngramms

trigrams = ngrames(mots_mo, 2)


