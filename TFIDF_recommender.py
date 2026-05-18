import json
import pandas as pd
from nltk.stem import PorterStemmer
import numpy as np
from  scipy.spatial.distance import cosine
from heapq import nsmallest,nlargest
import sys

# Read data from stdin
doc = sys.stdin.readline()
data = pd.read_csv('my_dataframe.csv')

count_vector={}
with open('my_dict.json', 'r') as f:
    count_vector = json.load(f)

Term_set=[]

with open('Term_set.json', 'r') as f:
    Term_set = json.load(f)

def tokenize(txt):
  txt=str(txt)
  txt = txt.replace(';',' ')
  return txt.split()

def porter_stemmer(words):
  stemmer = PorterStemmer()
  return [stemmer.stem(word) for word in words]

def user_count_vector(doc): 
    lst=[]
    doc=tokenize(doc)
    doc=porter_stemmer(doc)   
    for j in Term_set:
         lst.append(doc.count(j))
    return lst


def top_similar_doc_cosine(count_vec,doc,k=3):
        lst={}
        for i in count_vec:
            lst[i]=1-cosine(count_vec[i],user_count_vector(doc))
        top_similar_doc = nlargest(k, lst, key = lst.get)
        return lst,top_similar_doc

#doc='programming language;data type;scene model;inductive assertion;program performance;programming automation;programming languagesPrimitive operation;tape-bounded Turing acceptors;control structure;subgoal induction'
#doc=input("Please Enter keywords of your research interest:\n")
user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
rank=[]
user=[]
score=[]
kw=[]
count=1
for i in rec_adv:
               rank.append(count)
               user.append(i)
               score.append(user_score[i])
               a=data[data['n']==i]['t']
               for j in a:
                    k=" ".join(tokenize(j))
               kw.append(k)
               count+=1
df = {
                                    'Ranking': rank,
                                    'Name': user,
                                    'Similarity Score': score,
                                    'Keywords': kw
                                }
data_str = json.dumps(df)
print(data_str)
#print("Done")
with open('rec_result.json', 'w') as f:
    json.dump(df, f)
"""
for i in rec_adv:
  print(i, user_score[i])

for i in rec_adv:
  print(i)
  a=data[data['n']==i]['t']
  for j in a:
    print(" ".join(tokenize(j)))

"""

