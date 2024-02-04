#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Modification du dossier par défaut
import os
os.chdir('/Users/macbookair/Desktop/categorization/dataset')
#import file
import xlrd


# #### Importation des librairies necessaires

# In[63]:


import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import urllib
import io
from urllib.parse import urlencode, quote_plus
from urllib.request import Request, urlopen
import gzip
import json
from collections import Counter 
import nltk 
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# ### CLUSTERING DES DOCUMENTS

# #### Chargement des données
# 
# - Dans ce jeux de données on a effectué une transformation du dataframe de base de maniére à avoir les mots clés des articles sur une colonne dénommée keyword.
# - Ce qui nous donne le dataset chargé ci-dessous.

# In[64]:


dataset = pd.read_csv('datset2.csv',encoding='latin_1',sep = ',')
dataset.info()


# In[65]:


dataset.head(5)


# #### Transformation du dataframe
# - Ici nous transformons notre dataframe en dicttionnaire pour faciliter les opération suivantes.

# In[66]:


dict_df = dataset.to_dict()


# 
# - On recupéres les clés(titres) et les valeurs(keywords) sous forme de dictionaires

# In[67]:


#cles, vals = zip(*dict_df.items())
cles = list(dict_df.keys())
vals = list(dict_df.values())


# In[ ]:


print(cles)
print(vals)
print(vals[1])


# In[70]:


cles0, vals0 = list(vals[0].keys()),list(vals[0].values())
cles1, vals1 = list(vals[1].keys()),list(vals[1].values())


# 
# - On recupéres les clés(titres) et les valeurs(keywords) sous forme de liste

# In[72]:


titres, keywords = list(vals[1].keys()),list(vals[1].values())


# #### Création de la matrice document-termes 
# - utilisation de la librairie sklearn
# - ici les lignes représente les articles les colonnes les mots clés

# In[73]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(keywords)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
df['document'] = list(vals[0].keys())
df = df.set_index('document')


# - Affichage de quelques lignes et quelques colonnes de la matrice

# In[80]:


df[['3d','absolute','abstract','abstraction','abstractive','abuse','word','words','workers']].head()


# #### Réalisation du clustering
# - Ici nou générons la matrice des liens Z avec linkage
# - Génération et affichage du dendrogramme
# - Affichages du nombre de clusters obtenu selon le niveau de coupure t=5

# In[32]:


#librairies pour la CAH
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch

#générer la matrice des liens 
#Z = linkage(df,method='single',metric='jaccard') 
Z = linkage(df,method='ward',metric='euclidean')

# génération et affichage du dendrogramme
plt.figure(figsize=(20,12))
plt.title("CAH") 

dendrogram(Z,labels=list(vals[0].keys()),color_threshold=100)

plt.show() 

groupes_cah = sch.fcluster(Z,t=5,criterion='distance') 
print(np.unique(groupes_cah).size, "groupes constitués")


# - Nous remarquons 26 clusters avec un tel niveau de coupure

# #### Construction du dataframe avec les clusters
# - Ici on crée un dataframe des individus et leurs classes d'appartenance

# In[34]:


#index triés des groupes
import numpy as np
idg = np.argsort(groupes_cah)


# - affichage des observations et leurs groupes

# In[81]:


#affichage des observations et leurs groupes
dataf = pd.DataFrame(df.index[idg],groupes_cah[idg])
groupes = dataf
groupes['groupe'] = groupes.index
groupes.


# - Comptage du nombre d'individus par cluster

# In[ ]:


#nombre d'individus par clusters
groupes['groupe'].value_counts()


# #### Affichage des groupes sur le plan factoriel
# - Ici nous allons visualiser nos 26 clusters sur un plan factoriel 
# - avec un code couleur pour chaque groupe.

# In[82]:


colors = ['blue','lawngreen','red','indigo', 'aqua', 'yellow','orange','black','purple','pink',
          'beige','chocolate','coral','crimson','cyan','fuchsia','gold','indigo','green','lime',
          'magenta','navy','olive','plum','salmon','green']

numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]


# In[39]:


from sklearn.decomposition import PCA
#ACP
acp_subset = PCA(n_components=2).fit_transform(df)
#projeter dans le plan factoriel
#avec un code couleur selon le groupe
#remarquer le rôle de zip()
plt.figure(figsize=(15,10))
for couleur,k in zip(colors,numbers):
    plt.scatter(acp_subset[groupes_cah==k,0],acp_subset[groupes_cah==k,1],c=couleur)
    
#mettre les labels des points
#remarquer le rôle de enumerate()
for i,label in enumerate(df.index):
    plt.annotate(label,(acp_subset[i,0],acp_subset[i,1])) 
    
plt.show()


# In[83]:


# articles de chaque cluster ou groupe
groupes.groupby(['groupe','document']).count()


# #### Reconstitution du dataframe de base avec les clusters de chaque document
# - On crée la colonne titres et keywords sous forme de liste
# - en interrogeant le dataframe de base(dataset) et le dataframe groupes

# In[41]:


elements = list(groupes.document)
titres=[]
keywords = []
for i in elements:
    rowData = dataset.loc[i, : ]
    titres.append(rowData['article_title'])
    keywords.append(rowData['keyword'])


# In[84]:


# création de la colonne titre et keywords sur groupes
groupes['titre_article'], groupes['keywords'] = titres, keywords


# - Affichage du dataframe reconstitué

# In[87]:


groupes.head()


# ### LABELISATION DES ARTICLES D'UN CLUSTER
# #### Cas ou on se base sur les 10 premiers mots clés les plus recurrents des articles du cluster
# 
# Maitenant que nos clusters de documents sont construits, on va devoir attaquer la partie etiquettage. Le pricincipe utilisés est bien expliqué dans l'article.
# - Regroupper l'ensemble des keywords du clusters
# - Prendre les top 10 de keywords plus recurents
# - Construire des n-grammes(n=1...3) sur ces 10 mots clés
# - Interroger l'API Babelnet pour extraire les synsets, categories et domains.
# 
# Pour des contraintes techniques avec babelnet nous allons pas pouvoir executer le code sur l'ensemble des clusters en meme temps, car le nombre de requetes qu'on peut faire est limité(1000) pour la licence dont nous disposons. Du coup on va adopter une approche semi-automatique
# 
# - Cas du clusters 1;

# In[89]:


cluster = 1
cluster_kwlist = pd.DataFrame(groupes.groupby('groupe')['keywords'].apply(' '.join))
cluster_kwlist_group = cluster_kwlist[cluster_kwlist.index==cluster]


# In[92]:


cluster_kwlist.head(10)


# In[93]:


## Keywords du clusters 1
cluster_kwlist_group


# In[94]:


# comptage du nombre d'occurences par mots clés
split_it = cluster_kwlist_group['keywords'][cluster].split()
Counter = Counter(split_it)
nbr_mots = 10
most_occur = Counter.most_common(nbr_mots)
mots_mo = ' '.join([x[0] for x in most_occur])


# In[95]:


# Fonction de création des n-grammes
def ngrames(sentence,n):
    ngramms = []
    ngrm = nltk.ngrams(sentence.split(), n)
    for grm in ngrm:
        ngramms.append(list(grm))
    return ngramms


# #### Extraction des synonymes, catégories et dommaines

# In[96]:



# Initialisation
cat_all = {'cat':[],'mot':[]}
cat_all = pd.DataFrame(data=cat_all)
domain_all = {'domains':[],'ratio':[],'mot':[]}
domain_all = pd.DataFrame(data=domain_all)
ngrammes = [3,2,1]
for k in ngrammes:
    
    for i in ngrames(mots_mo, k):
        i = ' '.join(i)
        print(i)
        service_url = 'https://babelnet.io/v6/getSynsetIds'
        
        params = {
                'lemma' : i,
                'searchLang' : 'EN',
                'key'  : '53cc96fc-8b88-4ab4-8af8-b4e2870bac46'
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
            'key'  : '53cc96fc-8b88-4ab4-8af8-b4e2870bac46'
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


# ### Affichage des resultats
# 
# - Affichages des 10 premiers catégories obtenues

# In[98]:


cat_all.head(10)


#  - Affichages des 10 premiers domaines obtenues

# In[99]:


domain_all.head(10)


# - Croisement des catégories et des mots clés

# In[101]:


occ1.head(10)


# - Croisement des domaines et des mots clés

# In[102]:


occ2


# #################################################################################################################

# ####  Cas ou on se base sur les mots clés d'un articles du clusters
# - Ici on chosit le top 1 des articles de chaque clusters
# - On recupére les mots clés de l'article
# - On constitue les n-grammes à partir du mots clés
# - On intérroge BabelNet Api avec les trigrammes qui a tendance à données des domaines exactes et uniques
# - S'il ne retourne pas de resultats, on l'interroge avec les bigrammes, sinon avec les mots clés de l'articles pris un par un.

# In[111]:


# Recuperation des top1 article de chaque clusters(26 clusters ---> 26 articles)
top1_groupe = groupes.groupby(['groupe'])['groupe','titre_article','keywords'].apply(lambda x: x.nlargest(1, columns=['groupe']))
top1_groupe = top1_groupe[['groupe','titre_article','keywords']].set_index(top1_groupe['groupe'])
top1_groupe[['titre_article','keywords']]


# In[ ]:


# Recuperation des mots clés du clusters
keywords = top1_groupe.loc[1,:]['keywords']


# In[ ]:


trigrams = ngrames(keywords, 3)

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


# In[56]:


occ2


# In[57]:


trigrams


# In[58]:


domain_all


# In[ ]:




