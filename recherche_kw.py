# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:06:43 2021

@author: lamot
"""


import os
import pandas as pd
import urllib
import io
from urllib.parse import urlencode, quote_plus
from urllib.request import Request, urlopen
import gzip
import json
import nltk
import numpy as np

os.chdir("/Users/macbookair/Desktop/categorization/dataset")
data = pd.read_csv("pub_title.csv")

def ngrames(sentence,n):
    ngramms = []
    ngrm = nltk.ngrams(sentence.split(), n)
    for grm in ngrm:
        ngramms.append(list(grm))
    return ngramms

def bigrames(sentence):
    nltk_tokens = nltk.word_tokenize(sentence)  
    return list(nltk.bigrams(nltk_tokens))

#titre = '3D DYNAMIC EFFICIENT FACES MODELING REGISTRATION SPATIOTEMPORAL'
#titre = 'CO JUNE LOCATED PROCEEDINGS SETS TOOLS UK WORKSHOP'
# bigram
#titre = 'BASED DESCRIPTION LOGIC SET SUB SUP'
#titre = 'gender, medias, sport, hegemonic masculinity, emphasized feminity, editorial strategy'
#Mathematiquea et statistique (trigramme)
titre = 'Time-nonhomogeneous stochastic model process Partial differential equation Negative binomial distribution queue, Infinitely divisible distribution.'
#titre = '3D CONTAINING DISCOVERY IMAGES INSTANCES MODELING MULTIPLE OBJECT RGB SINGLE USING'
#titre = 'multi attention recurrent network human communication comprehension'
#titre = 'simulate brain signal create synthetic eeg data via neural base generative model improve ssvep classification'

keywords = ngrames(titre, 1)
bigrams = ngrames(titre, 2)
trigrams = ngrames(titre, 1)

cat_all = {'cat':[],'mot':[]}
cat_all = pd.DataFrame(data=cat_all)

domain_all = {'domains':[],'ratio':[],'mot':[]}
domain_all = pd.DataFrame(data=domain_all)
corbeill = ['MUSIC_SOUND_AND_DANCING',' MEDIA_AND_PRESS']

for i in trigrams:
    i = ' '.join(i)
    print(i)
    service_url = 'https://babelnet.io/v6/getSynsetIds'
    
    params = {
            'lemma' : i,
            'searchLang' : 'EN',
            'key'  : '1ed7054d-95e6-46f3-a86a-de79d654fb23'
    }
    
    url = service_url + '?' + urlencode(params)
    request = Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urlopen(request)

    if response.info().get('Content-Encoding') == 'gzip':
        buf = io.BytesIO(response.read())
        f = gzip.GzipFile(fileobj=buf)
        data_ids = json.loads(f.read())

        
    ids = [d['id'] for d in data_ids]
    cat = []
    domains = []
    ratio = []
    for j in ids:
        service_url = 'https://babelnet.io/v6/getSynset'
        
        params = {
        'id' : j,
        'key'  : '1ed7054d-95e6-46f3-a86a-de79d654fb23'
        }
        
        url = service_url + '?' + urlencode(params)
        request = Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urlopen(request)
        
        if response.info().get('Content-Encoding') == 'gzip':
            buf = io.BytesIO(response.read())
            f = gzip.GzipFile(fileobj=buf)
            data_cat = json.loads(f.read())
            cat = cat + data_cat['categories']
            domains=domains+list(data_cat['domains'].keys())
            ratio = ratio+list(data_cat['domains'].values())
            
    cat_art = [d['category'] for d in cat]
    domains_art = [d for d in domains]
    ratio_mot = [r for r in ratio]
    
    kw1 = [i]*len(cat_art)
    kw2 = [i]*len(domains_art)
    kw3 = [i]*len(ratio)
    
    df2 = {'cat':cat_art,'mot':kw1}
    df2 = pd.DataFrame(data=df2)
    cat_all = cat_all.append(df2)
    
    dfd = {'domains':domains_art,'ratio': ratio_mot, 'mot':kw2}
    dfd = pd.DataFrame(data=dfd)
    domain_all = domain_all.append(dfd)
   
occ1 = pd.crosstab(cat_all['cat'],cat_all['mot'])
occ2 = pd.crosstab(domain_all['domains'],domain_all['mot'])





'''
def getSynsetsIds(keywords):
    
    service_url = 'https://babelnet.io/v6/getSynsetIds'
    synsetsIdsBykeyword = dict()
    
    for i in keywords:
        i = ' '.join(i)
        params = {
                'lemma' : i,
                'searchLang' : 'EN',
                'key'  : '1ed7054d-95e6-46f3-a86a-de79d654fb23'
        }
        
        url = service_url + '?' + urlencode(params)
        request = Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urlopen(request)
    
        if response.info().get('Content-Encoding') == 'gzip':
            buf = io.BytesIO(response.read())
            f = gzip.GzipFile(fileobj=buf)
            data_ids = json.loads(f.read())
            synsetsIdsBykeyword[i] = [d['id'] for d in data_ids]
            
    return synsetsIdsBykeyword

def getCategoriesBySynsetId(ids):
    
    service_url = 'https://babelnet.io/v6/getSynset'
    categoriesById = dict()
    
    for id in ids:
        params = {
            'id' : id,
            'key'  : '8f612e9f-6b79-4efb-8770-77b4736c5eb5'
        }
        
        url = service_url + '?' + urlencode(params)
        request = Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urlopen(request)
            
        if response.info().get('Content-Encoding') == 'gzip':
            buf = io.BytesIO(response.read())
            f = gzip.GzipFile(fileobj=buf)
            data_cat = json.loads(f.read())
            categoriesById[id] = [d['category'] for d in data_cat['categories']]
            
    return categoriesById





#bigram = bigrames('reconstruction semantic model unorganised storehouses');
#ngram = ngrames('reconstruction semantic model unorganised storehouses', 2)
#detail human avatars monocular video

data_test = data[data['article_title']=='complete estimate question multi-task approach depth completion monocular depth estimation']
titre = data_test['article_title'][0]

keywords = ngrames(titre, 1)
bigrams = ngrames(titre, 2)
trigrams = ngrames(titre, 3)

cat_all = {'cat':[],'mot':[]}
cat_all = pd.DataFrame(data=cat_all)

Ids = getSynsetsIds(keywords)
key,synsetid = list(Ids.keys()),list(Ids.values())

cat = getCategoriesBySynsetId(synsetid)

'''
        

   
