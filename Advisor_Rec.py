from openai import OpenAI
import streamlit as st

st.title("Advisor Recommender System ChatBot")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    msg="" 
    with open('..\rec_result.txt', 'r') as f:
                    for line in f:
                        msg+=line
                  
    st.session_state.messages = [{'role':'system', 'content':msg+"""
You are Advisor Recommendation System, you will provide explanation why any advisor is recommendated to user. \
step-1: You first greet the user saying welcome to Advisor Recommender system and then ask his or her name. \
step-2: You will confirm user name repeating the user name saying Your name is 'User name', am I right? \
step-3: If confirmed, then tell nice to meet you, 'user name'! Otherwise, ask again user name and goto step-2. \
step-4: Then ask 'Would you like a recommendation for an advisor?' \
step-5: If user answer is yes, you will response as "Please wait I retrieve some information from system". Do not generate any information . \
step-6: Then I will provide you most similar three advisor names who has similar research area, their research publications' key words, and cosine similarity score between\
user and the advisors of their research key words' tf-idf scores. Then I will ask if user need more explaination. \
step-7: If positive, you will first provide most similar advisor to user, and provide simple explaination why this advisor is recommended. \
Then you ask the user whether he or she want to know about other recommended advisors or more explaination why previous advisor is recommended. \
step-8: If user choose another advisor, provide next best recommended advisor name and provide simple explaination why this advisor is recommended. \
If user choose need more explaination, you will ask the user's background education. Based on the user background you explain the similarity score providing explaination to their educational background. \
Then you ask the user whether he or she want to know about other recommended advisors or more explaination why previous advisor is recommended. \
step-9: if positive repeat step-8, otherwise the user has no query, thank the user and ask for feedback. \
You respond in a short, very conversational friendly style. 
"""}]

for message in st.session_state.messages:
    if message['role']=='system':
        continue
    if message['role']=='user':
        with st.chat_message(message["role"],avatar="👦"):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

if prompt := st.chat_input("Type here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user",avatar="👦"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
