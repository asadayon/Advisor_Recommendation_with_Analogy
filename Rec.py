import streamlit as st
import pandas as pd
import json
from nltk.stem import PorterStemmer
import numpy as np
from  scipy.spatial.distance import cosine
from heapq import nsmallest,nlargest
from openai import OpenAI
from st_aggrid import AgGrid, JsCode, GridOptionsBuilder
import mysql.connector
from mysql.connector import Error

st.set_page_config("Advisor Recommendation", page_icon=":book:")
data = pd.read_csv('updated_dataframe.csv')

count_vector={}
with open('my_dict.json', 'r') as f:
        count_vector = json.load(f)

Term_set=[]

with open('Term_set.json', 'r') as f:
        Term_set = json.load(f)

def insert_message(connection, role, content):
    cursor = connection.cursor()
    cursor.execute("INSERT INTO chat_messages (role, content) VALUES (%s, %s)", (role, content))
    connection.commit()
    
def connect_to_db():
    return mysql.connector.connect(
        host=st.secrets["HOST"],
        port=st.secrets["PORT"],
        user=st.secrets["USER"],
        password=st.secrets["PASSWORD"],
        database=st.secrets["DATABASE"]  # Replace with your actual database name
    )        
        

def tokenize(txt):
  txt=str(txt)
  txt = txt.replace(';',' ')
  txt = txt.replace(',',' ')
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


def cosine_recommender(doc):
    # Read data from stdin
    

    user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
    user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
    rank=[]
    user=[]
    score=[]
    kw=[]
    publication=[]
    affiliation=[]
    count=1
    for i in rec_adv:
                   rank.append(count)
                   user.append(i)
                   score.append(user_score[i])
                   a=data[data['n']==i]['t']
                   publication.append(data[data['n']==i]['paper_list'].values[0])
                   affiliation.append(data[data['n']==i]['affiliation'].values[0])
                 
                   for j in a: 
                        k=" ".join(tokenize(j))
                        #k=tokenize(j)
                   kw.append(k)
                   print(kw)
                   count+=1
    df = {
                                        'Ranking': rank,
                                        'Name': user,
                                        'Similarity Score': score,
                                        'Keywords': kw,
                                        'Publication':publication,
                                        'Affiliation':affiliation
                                    }
    data_str = json.dumps(df)
    
    #print("Done")
    with open('rec_result.json', 'w') as f:
        json.dump(df, f)
    return data_str

def load_dict(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def social_network(dic):
    rec=[]
    df={}
    r_name=[]
    r_t=[]
    lst=dic['Name']
    print(lst)
    kw=dic['Keywords']
    data=pd.read_csv("dataframe_id.csv")
    soc_id=load_dict('top3_id.json')
    soc_val=load_dict('top3_val.json')    
    for i in range(len(lst)):
        name=lst[i]
        key=kw[i]
        for index,row in data.iterrows():
            name2=row['n']
            idx=row['id']
            a=row['t']
            for j in a: 
                        k=" ".join(tokenize(j))

            if name2==name:
                if key==k:
                    print("Found ID: ",idx, " of ", name)
                    break
        if str(float(idx)) in soc_id:
            soc_i=soc_id[str(float(idx))]
            soc_v=soc_val[str(float(idx))]
            rec.append(soc_i[soc_v.index(min(soc_v))])
    for idx in rec:
        for index,row in data.iterrows():
            name2=data['n']
            idx2=data['id']
            a=data['t']
            for j in a: 
                        k=" ".join(tokenize(j))
            if idx==idx2:
                r_name.append(name2)
                r_t.append(k)
    df = {
                                        
                                        'Name': r_name,
                                        
                                        'Keywords': r_t
                                    }
    data_str = json.dumps(df)
    
    #print("Done")
    with open('soc_result.json', 'w') as f:
        json.dump(df, f)
    return data_str
        
        
            
            
            
            
        



def LDA(keywords):

    from gensim import corpora, models,  similarities
    import ast
    import string

    
    lda_model = models.LdaModel.load('lda_model.model')
    dictionary = corpora.Dictionary.load('dictionary.dict')
    index = similarities.MatrixSimilarity.load('index_file.index')
    rank=[]
    top=[]
    topic_words=[]
    topic_prob=[]
    names=[]
    sim=[]
    kw=[]
    publication=[]
    affiliation=[]


    #new_doc = ['end', 'user', 'parallel', 'program', 'liter', 'program', 'key', 'practic', 'program', 'style', 'end-us', 'softwar', 'engin', 'softwar', 'develop', 'end-us', 'program', 'situat', 'program', 'experi', 'program', 'standard']
    new_doc=keywords
    punc=''',;.'''
    for i in punc:
        if i in new_doc:
            new_doc.replace(i,' ')
    new_doc=new_doc.split()
    new_doc=porter_stemmer(new_doc)  
    
    new_doc_bow = dictionary.doc2bow(new_doc)

    new_doc_distribution = lda_model.get_document_topics(new_doc_bow)
    sorted_doc_topics = sorted(new_doc_distribution, key=lambda x: -x[1])
    top3_topics = sorted_doc_topics[:3]
    for topic, prob in top3_topics:
        print(f"Topic {topic} with probability {prob}")
        top_words = lda_model.show_topic(topic, 10)
        words_only = [word for word, pro in top_words]
        print(f"Top words for topic {topic}: {', '.join(words_only)}")
        top.append(topic)
        topic_words.append(words_only)
        topic_prob.append(prob)
        
        
        
        
    query_lda = lda_model[new_doc_bow]
    sims = index[query_lda]    
    sims_list = list(enumerate(sims))
    sorted_sims = sorted(sims_list, key=lambda item: -item[1])
    top3_documents = sorted_sims[:3]
    count=1
    for doc_position, score in top3_documents:
        print(f"Document id: {doc_position}, name: {data['n'][doc_position]} with similarity score: {score}")
        rank.append(count)
        names.append(data['n'][doc_position])
        a=data['t'][doc_position]
        publication.append(data['paper_list'][doc_position])
        affiliation.append(data['affiliation'][doc_position])
        #a=data[data['n']==i]['t']
        a=a.replace(";"," ")
                  
        kw.append(a)
        print(a)
        sim.append(score)
        count+=1
    df1 = {
                                        'LDA_rank': rank,
                                        'LDA_Name': names,
                                        'Score': sim,
                                        'Keywords_LDA': kw,
                                        'Publication':publication,
                                        'Affiliation':affiliation
                                       
                                    }
    #data_str1 = json.dumps(df1)
    
    #print("Done")
    #with open('LDA_rec_result.json', 'w') as f:
     #   json.dump(df1, f)
        
        
    df2 = {
                                        'Topic': top,
                                        'Words': topic_words,
                                        'Probability': topic_prob
                                       
                                    }
    #data_str2 = json.dumps(df2)
    
    #print("Done")
    #with open('LDA_topic_result.json', 'w') as f:
     #   json.dump(df2, f)
    return df1,df2
        
        



if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.clicked = True


st.title("Advisor Recommender System ")
st.write("This study aims to assess users' understanding of social recommendation explanations provided by a large language model (LLM) integrated into this System.  ")
name=st.text_input('Enter your Name')
keywords=st.text_input('Enter keywords of your reseach interest (i.e. Machine Learning, Natural Language Processing, Cybersecurity, Cryptography, Data Science, Human Computer Interaction, Robotics,  Software Engineering, Cloud Computing, etc)')

st.button('Submit', on_click=click_button)
data_dict={}
flag=0
if st.session_state.clicked:
    #with st.chat_message("Assistant"):
        st.session_state.clicked = False
        #st.write("Hello "+name+"! Please wait while we retrieve some information.")
        import time
        with st.spinner(text="Hello "+name+"! Please wait while we retrieve some information."):
            output=cosine_recommender(keywords)           
            data_dict = json.loads(output)
            
            lda1,lda2=LDA(keywords)
            #soc=social_network(data_dict)
            #soc = json.loads(soc)
            #lda1=json.loads(lda1)
            #lda2=json.loads(lda2)
            
            if "openai_model" not in st.session_state:
                st.session_state["openai_model"] = "gpt-3.5-turbo"
            if "flag" not in st.session_state:          
                st.session_state["flag"] = data_dict
                with open('rec_result.txt', 'w') as f:
                        msg="User name is "+ name+". User reseach interests are "+keywords+". Top 3 recommended advisor list based on Cosine similarity:\n"
                        for i in range(len(data_dict['Ranking'])):
                            msg+=str(i+1)+'. name: '+ data_dict['Name'][i]
                            msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                            msg+='. Keywords: '+data_dict['Keywords'][i]+'\n'
                        f.write(msg)
            if "lda1" not in st.session_state:
                st.session_state["lda1"] = lda1
                with open('rec_result.txt', 'a') as f:
                        msg="\nTop 3 recommended advisor list based on LDA Topic modeling:\n"
                        for i in range(len(lda1['LDA_rank'])):
                            msg+=str(i+1)+'. name: '+ lda1['LDA_Name'][i]
                            msg+='. Similarity score: '+str(lda1['Score'][i])
                            msg+='. Keywords: '+lda1['Keywords_LDA'][i]+'\n'
                        f.write(msg)
            if "lda2" not in st.session_state:
                st.session_state["lda2"] = lda2
                with open('rec_result.txt', 'a') as f:
                        msg="\nTop LDA Topic selected:\n"
                        for i in range(len(lda2['Topic'])):
                            msg+=str(i+1)+'. Topic id: '+ str(lda2['Topic'][i])
                            #msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                            msg+='. Keywords: '+" ".join(lda2['Words'][i])+'\n'
                        f.write(msg)
            
#             if "soc" not in st.session_state:
#                 st.session_state["soc"] = soc
#                 with open('rec_result.txt', 'a') as f:
#                         msg="\nTop recommended advisor list based on Social modeling:\n"
#                         for i in range(len(soc['Name'])):
#                             msg+=str(i+1)+'. name: '+ soc['Name'][i]
#                             msg+='. Keywords: '+soc['Keywords'][i]+'\n'
#                         f.write(msg)
            

            if "messages" not in st.session_state:
                msg="" 
                with open('rec_result.txt', 'r') as f:
                                for line in f:
                                    msg+=line
                                    
                prompt="""
            An Advisor Recommendation System where recommendations are generated based on two models: text similarity and LDA topic similarity between the user's research interests and those of potential advisors.
 Input: You are given a username and three recommended advisor names to the user based on each model. You are also provided with user research interests, and research interests of three recommended advisors. 
Objective: To guide users through understanding the reasoning behind advisor recommendations based on cosine similarity, and LDA topic similarity. The goal is to enhance users' technical comprehension of how their research interests align with those of the recommended advisors.
Expected Outcome: Users should gain a clear understanding of why specific advisors were recommended, with explanations tailored to their research interests. Users should be able to grasp the concepts of cosine similarity, topic similarity via LDA, and accurately interpret recommendation scores.
Do not provide overly technical jargon unless asked by the user. Do not give lengthy explanations; keep responses short, concise and user-friendly. Do not assume the user understands complex mathematical concepts; provide examples when necessary.
Model Output:
 
            """
                st.session_state.messages = [{'role':'system', 'content':prompt+msg}]
                #response="Welcome "+name+"! Would you like an explanation of your recommendation for advisors?"
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                response = client.chat.completions.create(
                        model=st.session_state["openai_model"],
                        messages=[
                            {"role": "system", "content": msg+prompt}
                        ]
                    )
                response=response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": response})
                connection = connect_to_db()
                insert_message(connection, "LLM", response)
                connection.close()
                
                
if "flag" in st.session_state:
    df1 = pd.DataFrame(st.session_state["flag"])
    df2 = pd.DataFrame(st.session_state["lda1"])
    df3 = pd.DataFrame(st.session_state["lda2"])
    
    #left_column, right_column = st.columns(2)
    left_column, right_column = st.tabs(["Text Similarity", "Topic Similarity"])
    with left_column:
        df1_new = df1                 
        df1_new = df1_new.to_dict(orient='records')
        st.write("Top 3 recommended advisor based on Text Similarity of keywords:")
        #st.dataframe(df1_new, hide_index=True)
        columnDefs = [
          {
            'headerName': "Ranking",
            'field': "Ranking",
            # here the Athlete column will tooltip the Country value
            'tooltipField': "Ranking",
            'headerTooltip': "Advisor ranking based on cosine similarity",
            'width': 10, 
            
          },
            {
            'field': "Name",
            'tooltipValueGetter': JsCode("""function(p) {return "Paper List: \n"+p.data.Publication}"""),
            'headerTooltip': "Advisor Information",
            'width': 20, 
          },
          {
            'field': "Affiliation",
            'tooltipValueGetter': JsCode("""function(p) {return " Affiliation: \n"+p.data.Affiliation}"""),
            'headerTooltip': "Advisor Affiliation",
            'width': 120, 
          },
          
          ];
        gridOptions =  {
              'defaultColDef': {
                'flex': 1,
                
                
                
              },
              'rowData': df1_new,
              'columnDefs': columnDefs,
              'tooltipShowDelay': 200,
            };
        AgGrid(None, gridOptions,  height = 120,allow_unsafe_jscode=True)

    with right_column:
        st.write("Top 3 recommended advisor based on LDA Topic Similarity of 30 topics:")
        df2_new = df2
        df2_new = df2_new.to_dict(orient='records')
        #st.dataframe(df2_new,hide_index=True, column_config={
        #"LDA_rank": "Ranking","LDA_Name": "Name"})
        columnDefs = [
          {
            'headerName': "Ranking",
            'field': "LDA_rank",
            # here the Athlete column will tooltip the Country value
            'tooltipField': "LDA_rank",
            'headerTooltip': "Advisor ranking based on cosine similarity",
          },
            {'headerName': "Name",
            'field': "LDA_Name",
            'tooltipValueGetter': JsCode("""function(p) {return "Paper List: \n"+p.data.Publication}"""),
            'headerTooltip': "Advisor Information",
            'width': 20, 
          },
          {
            'field': "Affiliation",
            'tooltipValueGetter': JsCode("""function(p) {return " Affiliation: \n"+p.data.Affiliation}"""),
            'headerTooltip': "Advisor Affiliation",
            'width': 120, 
          },];
        gridOptions =  {
              'defaultColDef': {
                'flex': 1,
                'minWidth': 100,
              },
              'rowData': df2_new,
              'columnDefs': columnDefs,
              'tooltipShowDelay': 500,
            };
        AgGrid(None, gridOptions,  height = 120,allow_unsafe_jscode=True)

    

    
            



               




