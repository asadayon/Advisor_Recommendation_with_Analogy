import streamlit as st
import pandas as pd
import json
from nltk.stem import PorterStemmer
import numpy as np
from heapq import nsmallest,nlargest
from openai import OpenAI
from st_aggrid import AgGrid, JsCode, GridOptionsBuilder
import pickle
import pandas as pd
import requests
import random
import joblib
from PreQuiz.quiz import load_questions
import time, uuid
from datetime import datetime, timedelta
from supabase import create_client

st.set_page_config("Advisor Recommendation", page_icon=":book:")
data = pd.read_csv('updated_dataframe.csv')
lda_model = joblib.load('lda_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
doc_topic_matrix = joblib.load('doc_topic_matrix.pkl')
options = ["software engineering", "software process", "software system", "software quality", "design debt", "case studies", "software development", "software evolution", "online communities", "websites", "web pages", "related websites", "web spam", "web communities", "web mining", "online community analysis", "spammy website networks", "rescue robots", "autonomous mobile robots", "autonomous mode", "tele-operation mode", "multiple robots", "mobile robot", "proposed system", "mobile applications", "mobile devices", "smart phones", "mobile Internet devices", "context information", "resource-constrained mobile devices", "mobile users", "mobile phone", "mobile devices adaptive"]
API_URL= st.secrets["URL"]
MODEL   = st.secrets["MODEL"] 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL   = st.secrets["OPENAI_MODEL"] 
SUPABASE_URL = st.secrets["SUP_URL"]
SUPABASE_KEY = st.secrets["SUP_KEY"]
COOLDOWN_TIME_LONG = 5
COOLDOWN_TIME_SHORT = 2
NO_COOLDOWN = 0

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


        
def start_session(user_name, scenario, version):
    """
    Start a new session for a given user and scenario.
    Inserts a row into the 'session' table.
    Returns the new session info (including generated session_id).
    """
    if not st.session_state.session_id:
        resp = (
            supabase.table("session")
            .insert({
                "user_name": user_name,
                "scenario": scenario,
                "version": version
            })
            .execute()
        )
        st.session_state["session_id"] = resp.data[0]["session_id"]


def log_chat_message(role, content):
    """
    Insert a chat message into 'chat_message'.
    If turn_number is None, it uses:
      - next_turn() for role == 'user'
      - current_turn() for role == 'assistant'
    """
    sid = st.session_state["session_id"]

    payload = {
        "session_id": sid,
        "role": role,                   # 'user' | 'assistant' | 'system'
        "content": content
    }

    return supabase.table("chat_message").insert(payload).execute()


count_vector={}
with open('my_dict.json', 'r') as f:
        count_vector = json.load(f)

Term_set=[]



with open('Term_set.json', 'r') as f:
        Term_set = json.load(f)
        
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
        
def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)  # add epsilon to avoid divide by zero

def top_similar_doc_cosine(count_vec,doc,k=3):
        lst={}
        for i in count_vec:
            lst[i] = cosine_similarity(count_vec[i], user_count_vector(doc))
        top_similar_doc = nlargest(k, lst, key = lst.get)
        return lst,top_similar_doc
def render_spacer():
    st.markdown("""
    <div style='min-height: 250px; overflow-y: auto; padding: 10px;'>
                <h1></h1>
    </div>
    """, unsafe_allow_html=True)
def reset_lock_timer():
    keys_to_delete = [
        key for key in st.session_state.keys() 
        if key.endswith("_done") or key.endswith("_end_time") or key == "unlock_time"
    ]
    for key in keys_to_delete:
        del st.session_state[key]

def render_back_button(page):
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Home"):
            reset_common_state()
            reset_lock_timer()
            reset_prequiz_states()
            if page == "v1":
                st.session_state.followup_idx = 1
            st.session_state.page = "home"
            st.rerun()


def reset_common_state():
    reset_ai_state()
    st.session_state.show_explain_option = False
    st.session_state.session_id=None


def reset_ai_state():
    st.session_state.initial_prompt_sent = False
    st.session_state.chat_history = []
    st.session_state.chat_html = ""
    st.session_state.explain_clicked = False

def reset_prequiz_states():
    keys_to_clear = [
        "v2_quiz_index",
        "v2_selected_options",
        "v2_quiz_done",
        "v2_chat_history_per_q",
        "v2_sent_system_prompt",
        "v2_initial_radio_set",
        "v2_input_used",
        "v2_quiz_questions"
    ]
    keys_to_clear += ["final_chat_history", "final_streaming", "v2_show_final_chat"]

    # Also remove any selected option keys per question
    for key in list(st.session_state.keys()):
        if key.startswith("selected_option_q_") or key.startswith("option_radio_q_") or key.startswith("form_q_") or key.startswith("input_q_"):
            keys_to_clear.append(key)

    for key in keys_to_clear:
        st.session_state.pop(key, None)
def initialize_v2():
            st.session_state.v2_quiz_index = 0
            st.session_state.v2_selected_options = []
            st.session_state.v2_quiz_done = False
            st.session_state.v2_chat_history_per_q = {}
            st.session_state.v2_sent_system_prompt = {}
            st.session_state.v2_initial_radio_set = {}
            st.session_state.v2_input_used = {}
            


def chat_stream(provider="openai"):
    """
    provider: "ollama" or "openai"
    Streams assistant response token-by-token.
    """

    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    if provider == "ollama":
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": True
        }

        with requests.post(
            API_URL,
            headers=headers,
            json=payload,
            stream=True
        ) as r:
            r.raise_for_status()

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue

                data = json.loads(line)

                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]

                if data.get("done"):
                    break

    elif provider == "openai":
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": OPENAI_MODEL,
            "input": messages,
            "stream": True
        }

        with requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            stream=True
        ) as r:
            r.raise_for_status()

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue

                # OpenAI uses Server-Sent Events: lines start with "data: "
                if not line.startswith("data: "):
                    continue

                data_str = line[len("data: "):]

                if data_str == "[DONE]":
                    break

                data = json.loads(data_str)

                if data.get("type") == "response.output_text.delta":
                    yield data.get("delta", "")

                if data.get("type") == "response.completed":
                    break

def stream_llm_api(history, provider="openai"):
    """
    Streams assistant response from either Ollama or OpenAI API.

    Args:
        history: List of chat messages, e.g.
                 [{"role": "user", "content": "Hello"}]
        provider: "ollama" or "openai"

    Yields:
        Text chunks in real time for Streamlit display.
    """
    print("stream_llm_api")
    try:
        if provider == "ollama":
            yield from stream_ollama(history)

        elif provider == "openai":
            print("stream_llm_api->stream_openai")
            yield from stream_openai(history)

        else:
            raise ValueError("provider must be either 'ollama' or 'openai'")

    except requests.RequestException as e:
        st.error(f"LLM stream error: {e}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")


def stream_ollama(history):
    """
    Stream response from Ollama /api/chat endpoint.
    """
    
    payload = {
        "model": MODEL,
        "messages": history,
        "stream": True
    }

    headers = {
        "Content-Type": "application/json"
    }

    with requests.post(
        API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60
    ) as resp:

        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                obj = json.loads(line)

                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    yield chunk

                if obj.get("done"):
                    break

            except json.JSONDecodeError as parse_err:
                print("⚠️ Ollama chunk parsing error:", parse_err)
                continue


def stream_openai(history):
    """
    Stream response from OpenAI Responses API.
    """

    payload = {
        "model": OPENAI_MODEL,
        "input": history,
        "stream": True
    }
    print("stream_openai")
    print("payload: ",payload)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    with requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        stream=True,
        timeout=60
    ) as resp:

        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            # OpenAI streaming uses Server-Sent Events.
            if not line.startswith("data: "):
                continue

            data_str = line[len("data: "):]

            if data_str == "[DONE]":
                break

            try:
                obj = json.loads(data_str)

                if obj.get("type") == "response.output_text.delta":
                    chunk = obj.get("delta", "")
                    if chunk:
                        yield chunk

                if obj.get("type") == "response.completed":
                    break

            except json.JSONDecodeError as parse_err:
                print("⚠️ OpenAI chunk parsing error:", parse_err)
                continue

CORE_SYSTEM_KNOWLEDGE = """
                        Our system is designed to help prospective graduate students find suitable research advisors by matching them based on shared research interests and publications. The system uses two models: a **Text Similarity Model** and a **Topic Similarity Model**, each generating the top three advisor recommendations based on the user’s input keywords.
                        
                        1. **Text Similarity Model (Cosine Similarity):**
                           - Inputs: Research keywords provided by the user.
                           - Each advisor’s research profile is represented as a numerical count vector of publication keywords.
                           - Cosine similarity is calculated between the user’s keyword vector and each advisor’s vector.
                           - Output: Top 3 advisors with the highest similarity scores (range: 0 to 1), where values closer to 1 indicate stronger alignment.
                        
                        2. **Topic Similarity Model (LDA Topic Modeling):**
                           - Inputs: User’s research keywords mapped to 30 predefined LDA topics.
                           - Each advisor has a topic distribution profile learned from their publication data.
                           - The similarity between the user’s topic vector and each advisor’s topic profile is computed.
                           - Output: Top 3 advisors with the most similar topic distributions.
                           
                           Results are displayed in two tabs: one for Text Similarity and one for Topic Similarity, each showing advisors’ names, affiliations, and publication details. The recommendations aim to foster meaningful academic collaborations by aligning students with advisors whose research interests are most compatible.
                        """

def make_system_prompt( selected_topic, scenario,core_system_knowledge=CORE_SYSTEM_KNOWLEDGE):
    """
    Construct the system prompt explaining the two-model 
    architecture and the specific inputs/outputs for this session.
    """
    data_dict=st.session_state["cosine"]
    lda1=st.session_state["lda1"]
    lda2=st.session_state["lda2"]
    return f"""
            You are an AI-powered academic advisor chatbot designed to explain the reasoning behind advisor recommendations generated by a machine learning system. Your goal is to help prospective graduate students understand how their research interests align with those of faculty members based on two recommendation models.

            
            {core_system_knowledge}

            System Inputs and Outputs for This Session:

            Input:
             ## Current Student Context
        - Student Scenario: {scenario}
        - User selected topics: {selected_topic}
        - Top 3 recommended advisor list based on Cosine similarity:: 
            1. Name: {data_dict['Name'][0]}; Cosine similarity score: {data_dict['Similarity Score'][0]};  Keywords: {data_dict['Keywords'][0]}; Publication: {data_dict['Publication'][0]}; Affiliaiton: {data_dict['Affiliation'][0]}
            2. Name: {data_dict['Name'][1]}; Cosine similarity score: {data_dict['Similarity Score'][1]};  Keywords: {data_dict['Keywords'][1]}; Publication: {data_dict['Publication'][1]}; Affiliaiton: {data_dict['Affiliation'][1]}
            3. Name: {data_dict['Name'][2]}; Cosine similarity score: {data_dict['Similarity Score'][2]};  Keywords: {data_dict['Keywords'][2]}; Publication: {data_dict['Publication'][2]}; Affiliaiton: {data_dict['Affiliation'][2]}
        - Top 3 recommended advisor list based on LDA Topic modeling:: 
            1. Name: {lda1['LDA_Name'][0]}; Cosine similarity score: {lda1['Score'][0]};  Keywords: {lda1['Keywords_LDA'][0]}; Publication: {lda1['Publication'][0]}; Affiliaiton: {lda1['Affiliation'][0]}
            2. Name: {lda1['LDA_Name'][1]}; Cosine similarity score: {lda1['Score'][1]};  Keywords: {lda1['Keywords_LDA'][1]}; Publication: {lda1['Publication'][1]}; Affiliaiton: {lda1['Affiliation'][1]}
            3. Name: {lda1['LDA_Name'][2]}; Cosine similarity score: {lda1['Score'][2]};  Keywords: {lda1['Keywords_LDA'][2]}; Publication: {lda1['Publication'][2]}; Affiliaiton: {lda1['Affiliation'][2]}
        - Top LDA Topic selected:
                Topic id: {lda2['Topic'][0]}
                Keywords: {lda2['Words'][0]}
        ---

            **Expected Outcome:**
Help users interpret why these advisors were recommended, how closely their research interests align, and how changes in keywords might affect the results. Word counts more than 200.

**Guidelines:**
- Provide explanations in details why the advisors are recommended.
- You can answer both **general** and **scenario-specific** questions.

**For General Questions** (e.g., *"How does the system work?"*):
- Tell a brief of how advisors are recommended.
- Explain both models:
  • How keyword similarity (cosine similarity) works to recommend advisors.
  • How LDA groups keywords into research themes and compares distributions.
- Explain why using both models gives a more robust match.

**For Scenario-Specific Questions** (e.g., *"How the top advisors from both models recommended?"*):
- Show which terms contributed to high similarity. Provide example such as [kw1, kw2,..] to vector using users individual research keywords. Create vector like [1, 0, 1, 0] and Show a dot product calculation. Then a similarity score example using dot product.
- Show LDA based top selected topic and how advisor is selected from the topic.
- Explain how the user’s keywords closely matched the advisor’s keywords or topics.
- Mention concrete alignment in research themes.
- Highlight key differences between text vs. topic model rankings.
- Clarify what the similarity scores mean and that a lower score can still be meaningful in niche areas.

When asked **what-if scenarios** or asked for result explanations: (like: *"What if User had [different keywords]?"*, *Can you explain the results?*):
- Provide Feature-Based Explanation: Explains how users individual research keywords contributed to the results. Provide example using users individual research keywords contributes from rank 1 to rank 3 similar advisor.
- Counterfactual-Based Explanation: Shows how changing reseach keywords would alter predictions. Provide example using users individual research keywords change can make rank 3 to rank 1.

You are now ready to answer the user’s questions about their recommended graduate advisors.
        """.strip()


def make_quiz_system_prompt(question, options, correct_index, selected_topic, scenario, core_system_knowledge=CORE_SYSTEM_KNOWLEDGE):

    
    formatted_options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    data_dict=st.session_state["cosine"]
    lda1=st.session_state["lda1"]
    lda2=st.session_state["lda2"]
    prompt = f"""
        You are acting as an **explanation assistant** for a Grad Student Advisor recommender system.
        You have full internal knowledge of how the system works.

        ---
        ## System Knowledge
        {core_system_knowledge}

        ---
        ## Current Student Context
        - Student Scenario: {scenario}
        - User selected topics: {selected_topic}
        - Top 3 recommended advisor list based on Cosine similarity:: 
            1. Name: {data_dict['Name'][0]}; Cosine similarity score: {data_dict['Similarity Score'][0]};  Keywords: {data_dict['Keywords'][0]}; Publication: {data_dict['Publication'][0]}; Affiliaiton: {data_dict['Affiliation'][0]}
            2. Name: {data_dict['Name'][1]}; Cosine similarity score: {data_dict['Similarity Score'][1]};  Keywords: {data_dict['Keywords'][1]}; Publication: {data_dict['Publication'][1]}; Affiliaiton: {data_dict['Affiliation'][1]}
            3. Name: {data_dict['Name'][2]}; Cosine similarity score: {data_dict['Similarity Score'][2]};  Keywords: {data_dict['Keywords'][2]}; Publication: {data_dict['Publication'][2]}; Affiliaiton: {data_dict['Affiliation'][2]}
        - Top 3 recommended advisor list based on LDA Topic modeling:: 
            1. Name: {lda1['LDA_Name'][0]}; Cosine similarity score: {lda1['Score'][0]};  Keywords: {lda1['Keywords_LDA'][0]}; Publication: {lda1['Publication'][0]}; Affiliaiton: {lda1['Affiliation'][0]}
            2. Name: {lda1['LDA_Name'][1]}; Cosine similarity score: {lda1['Score'][1]};  Keywords: {lda1['Keywords_LDA'][1]}; Publication: {lda1['Publication'][1]}; Affiliaiton: {lda1['Affiliation'][1]}
            3. Name: {lda1['LDA_Name'][2]}; Cosine similarity score: {lda1['Score'][2]};  Keywords: {lda1['Keywords_LDA'][2]}; Publication: {lda1['Publication'][2]}; Affiliaiton: {lda1['Affiliation'][2]}
        - Top LDA Topic selected:
                Topic id: {lda2['Topic'][0]}
                Keywords: {lda2['Words'][0]}
        ---
        ## Current Quiz Task
        The user is working through a **pre-quiz** designed to prepare them for a longer comprehension test.
        They are answering the following question:

        "{question}"

        Options:
        {formatted_options}

        The correct answer is **option {str(correct_index)}**. The user will select one of the options and you will provide feedback based on their selection.

        ---
        ## Special Instructions
        - If the user says "Option [OPTION NUMBER] has been selected", you should respond as follows:'
            - If the user selects the correct answer, explain why it is correct and if not done yet, explain about the system too.
            - If they select an incorrect answer, provide explanation on why it was wrong and guide them to the correct reasoning.
        - The first time the user selects an option, you should give a brief explaination of the system and how it works and then begin to answer as per the specific option selected.
        
        ## Your Role & Style Guide
        - Your main goal is to help the user **understand the system reasoning** and explain why the selected options are either correct or not.
        - Encourage step-by-step reasoning based on the system's recommendations, similarity scores, and reasoning logic.
        - Avoid generic advice; always tie reasoning back to **how this specific system** would think.
        - Keep explanations **short, targeted, and context-aware** - no long lectures.
        - If the user asks follow up questions and seems unsure, ask small guiding questions rather than giving away the answer if they have not selected the correct option yet.
        - When explaining, use simple language and avoid technical jargon unless the user asks for it.

        Respond in a **supportive and educational tone**.
            """
    
    return prompt.strip()

def render_v2_quiz_flow(questions, idx, scenario):
    question = questions[idx]
    qid = idx+1
    st.markdown(f"#### Q{idx+1}: {question['question']}")
    options = question["options"]

    selected_option_key = f"selected_option_q_{qid}"
    radio_key = f"option_radio_q_{qid}"

    if selected_option_key not in st.session_state:
        st.session_state[selected_option_key] = None

    prev_selected = st.session_state[selected_option_key]
    st.markdown("__*Please select one of the options to know more about it.*__")
        
    selected = st.radio(
        "Select your answer:",
        options,
        index=options.index(prev_selected) if prev_selected in options else None,
        key=radio_key
    )

    if selected != prev_selected:
        st.session_state[selected_option_key] = selected
        chosen_index = options.index(selected) + 1 if selected in options else None
        if chosen_index is not None:
            selection_msg = f"Option {chosen_index} has been selected."

            # Ensure chat history for this question exists
            if qid not in st.session_state.v2_chat_history_per_q:
                st.session_state.v2_chat_history_per_q[qid] = []
            chat_history = st.session_state.v2_chat_history_per_q[qid]

            # Append to history
            chat_history.append({"role": "user", "content": selection_msg})

            # Show immediately
            with st.chat_message("user"):
                st.markdown(selection_msg)

            # Trigger LLM streaming next run
            streaming_flag_key = f"v2_is_streaming_{qid}"
            st.session_state[streaming_flag_key] = True

            st.rerun()


    correct = int(question['answer'])
    chosen = options.index(selected) + 1 if selected in options else None
 

    if chosen:
        if chosen == correct:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect.")

    # Initialize quiz session state dicts if needed
    if "v2_chat_history_per_q" not in st.session_state:
        st.session_state.v2_chat_history_per_q = {}
    if "v2_sent_system_prompt" not in st.session_state:
        st.session_state.v2_sent_system_prompt = {}
    if "v2_input_used" not in st.session_state:
        st.session_state.v2_input_used = {}

    if qid not in st.session_state.v2_chat_history_per_q:
        st.session_state.v2_chat_history_per_q[qid] = []

    chat_history = st.session_state.v2_chat_history_per_q[qid]

    if qid not in st.session_state.v2_sent_system_prompt:
        system_prompt = make_quiz_system_prompt(
            question['question'], options, correct,
            st.session_state.selected_keywords,
            scenario
        )
        chat_history.append({"role": "system", "content": system_prompt})
        chat_history.append({"role": "dummy", "content": "Option 1 has been selected."})
        st.session_state.v2_sent_system_prompt[qid] = True

    # Render all messages excluding system
    
    for msg in chat_history:
        if msg["role"] == "system" :
                continue
        if msg["role"] == "dummy" :
                continue       
        with st.chat_message(msg['role']):
                st.markdown(msg['content'])

    # --- Add streaming flag init per question ---
    streaming_flag_key = f"v2_is_streaming_{qid}"
    if streaming_flag_key not in st.session_state:
        st.session_state[streaming_flag_key] = False

    # Only render form and collect input if this is the active quiz question AND not currently streaming
    
    if idx == st.session_state.v2_quiz_index:
        if st.session_state[streaming_flag_key]:
            try:
                with st.chat_message("assistant"):
                    response_container = st.empty()
                    assistant_text = ""
                    for chunk in stream_llm_api(chat_history):
                        assistant_text += chunk
                        response_container.markdown(assistant_text + "▌")
                    response_container.markdown(assistant_text)
                    chat_history.append({"role": "assistant", "content": assistant_text})
                    log_chat_message("assistant", assistant_text)
                st.session_state[streaming_flag_key] = False  # done streaming
                st.rerun()  # rerun so form can show next run
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"LLM error: {e}")
                st.session_state[streaming_flag_key] = False
        else:
            if st.session_state.get("v2_show_final_chat", False):
                # If final chat is showing, skip rendering per-question input form
                pass
            else:
                st.markdown("__*Please use the chatbot below to learn more.*__")

                # Not streaming → show form to collect user input
                user_input = countdown_with_form(
                    message="Please read the text question carefully before answering.",
                    duration_sec=st.session_state.get("NO_COOLDOWN", NO_COOLDOWN),
                    form_key=f"form_q_{qid}",
                    input_key=f"input_q_{qid}"
                )

                if user_input:
                    chat_history.append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    # Set streaming flag to true to trigger streaming on next rerun
                    st.session_state[streaming_flag_key] = True
                    st.rerun()

        # Show Next / Finish button logic
        if st.session_state.v2_quiz_index < len(questions) - 1:
            # For all but last question, show normal Next button
            if st.button("Next Question", key=f"next_btn_{idx}"):
                st.session_state.v2_selected_options.append({
                    "question_id": idx+1,
                    "selected": chosen,
                    "correct": correct
                })
                st.session_state.v2_quiz_index += 1
                st.rerun()

        else:
            # Last question: control when to show final chatbot form with a flag
            if not st.session_state.get("v2_show_final_chat", False):
                # Show a "Finish Quiz" button first
                if st.button("Finish Quiz", key=f"finish_btn_{idx}"):
                    # Save last question answer before finishing
                    st.session_state.v2_selected_options.append({
                        "question_id": idx+1,
                        "selected": chosen,
                        "correct": correct
                    })
                    st.session_state.v2_show_final_chat = True
                    st.rerun()

            else:
                # Show final chatbot input after "Finish Quiz" pressed
                st.markdown("---")
                st.markdown("#### 📝 Do you have any other questions?")
                system_prompt = make_system_prompt(
                    st.session_state.selected_keywords,
                    scenario,
                    core_system_knowledge=CORE_SYSTEM_KNOWLEDGE
                )
                if "final_chat_history" not in st.session_state:
                    st.session_state.final_chat_history = [
                        {"role": "system", "content": system_prompt}
                    ]
                if "final_streaming" not in st.session_state:
                    st.session_state.final_streaming = False

                # Render final chat transcript so far
                for msg in st.session_state.final_chat_history:
                    if msg["role"] == "system":
                        continue
                    if msg["role"] == "dummy":
                        continue

                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Stream assistant if streaming flag set
                if st.session_state.final_streaming:
                    with st.chat_message("assistant"):
                        response_container = st.empty()
                        assistant_text = ""
                        for chunk in stream_llm_api(st.session_state.final_chat_history):
                            assistant_text += chunk
                            response_container.markdown(assistant_text + "▌")
                        response_container.markdown(assistant_text)

                    st.session_state.final_streaming = False
                    st.session_state.final_chat_history.append({"role": "assistant", "content": assistant_text})
                    st.rerun()

                else:
                    # Show freeform input form
                    user_input = countdown_with_form(
                        message="Please wait",
                        duration_sec=st.session_state.get("NO_COOLDOWN", NO_COOLDOWN),
                        form_key="final_freeform_form",
                        input_key="final_freeform_input"
                    )

                    if user_input:
                        st.session_state.final_chat_history.append({"role": "user", "content": user_input})
                        with st.chat_message("user"):
                            st.markdown(user_input)
                        st.session_state.final_streaming = True
                        st.rerun()

    st.session_state.v2_chat_history_per_q[qid] = chat_history

def load_prequiz_questions(scenario):
    if "v2_quiz_questions" not in st.session_state:
        student_name = ("").join(scenario.split(" ")[:2])
        print(student_name)
        resp = load_questions(student_name)
        print(resp)
        st.session_state.v2_quiz_questions = resp

    return st.session_state.v2_quiz_questions

def render_v2(scenario):
        questions = load_prequiz_questions(scenario)
        idx = st.session_state.get("v2_quiz_index", 0)
        total = len(questions)
        if idx < total:
            for i in range(idx + 1):
                render_v2_quiz_flow(questions, i, scenario)

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




def countdown_component_html(message, duration_sec, reveal_html):
    # Initialize unlock_time only when not already set
    if "unlock_time" not in st.session_state:
        st.session_state.unlock_time = datetime.now() + timedelta(seconds=duration_sec)

    remaining = int((st.session_state.unlock_time - datetime.now()).total_seconds())
    
    html_code = f"""
    <div style="font-weight:bold;font-size:16px;">
        <span id="timer">{message} — {remaining//60:02d}:{remaining%60:02d}</span>
    </div>

    <div id="reveal-section" style="display:none; margin-top:10px;">
        {reveal_html}
    </div>

    <script>
    var seconds = {remaining};
    var timerElement = document.getElementById("timer");
    var revealSection = document.getElementById("reveal-section");
    var countdown = setInterval(function(){{
        if (seconds > 0) {{
            seconds--;
            var mins = Math.floor(seconds/60);
            var secs = seconds % 60;
            timerElement.innerHTML = "{message} — " + 
                (mins<10?"0":"") + mins + ":" + (secs<10?"0":"") + secs;
        }} else {{
            clearInterval(countdown);
            timerElement.innerHTML = "You can now proceed!";
            revealSection.style.display = "block";
        }}
    }}, 1000);
    </script>
    """

    st.components.v1.html(html_code, height=120)
    
def countdown_with_button(message, duration_sec, button_label, button_key):
    # Initialize countdown state
    if f"{button_key}_done" not in st.session_state:
        st.session_state[f"{button_key}_done"] = False

    if not st.session_state[f"{button_key}_done"]:
        placeholder = st.empty()
        for remaining in range(duration_sec, 0, -1):
            mins, secs = divmod(remaining, 60)
            placeholder.markdown(f"**{message} — {mins:02d}:{secs:02d}**")
            time.sleep(1)
        placeholder.empty()
        st.session_state[f"{button_key}_done"] = True

    # Show button only after countdown done
    return st.button(button_label, key=button_key)


    
def countdown_with_form(message, duration_sec, form_key, input_key, submit_label="➤"):
    """
    Shows a countdown before revealing a form with text input + submit.
    Returns the user input if submitted, else None.
    """
    if f"{form_key}_done" not in st.session_state:
        st.session_state[f"{form_key}_done"] = False

    if not st.session_state[f"{form_key}_done"]:
        placeholder = st.empty()
        for remaining in range(duration_sec, 0, -1):
            mins, secs = divmod(remaining, 60)
            placeholder.markdown(f"**{message} - {mins:02d}:{secs:02d}**")
            time.sleep(1)
        placeholder.empty()
        st.session_state[f"{form_key}_done"] = True

    # Show form after countdown done
    if st.session_state[f"{form_key}_done"]:
        with st.form(form_key, clear_on_submit=True):
            cols = st.columns([4, 0.5])
            user_input = cols[0].text_input("", key=input_key, label_visibility="collapsed")
            send = cols[1].form_submit_button(submit_label)
            if send and user_input:
                return user_input
    return None

def load_dict(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def LDA(keywords):
    rank, top, topic_words, topic_prob = [], [], [], []
    names, sim, kw, publication, affiliation = [], [], [], [], []
    from sklearn.metrics.pairwise import cosine_similarity as cosim

    # Preprocess user keywords
    new_doc = porter_stemmer(tokenize(keywords))
    new_doc_text = " ".join(new_doc)
    new_doc_vector = vectorizer.transform([new_doc_text])

    # Topic distribution
    topic_distribution = lda_model.transform(new_doc_vector)[0]
    top3_indices = topic_distribution.argsort()[-3:][::-1]
    # Similarity with existing documents
    new_topic_matrix = lda_model.transform(new_doc_vector)
    similarities = cosim(new_topic_matrix, doc_topic_matrix)[0]
    sorted_sims = similarities.argsort()[-3:][::-1]



    for topic in top3_indices:
        prob = topic_distribution[topic]
        print(f"Topic {topic} with probability {prob:.4f}")
        topic_terms = lda_model.components_[topic]
        top_words_idx = topic_terms.argsort()[-10:][::-1]
        words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
        print(f"Top words for topic {topic}: {', '.join(words)}")
        top.append(topic)
        topic_words.append(words)
        topic_prob.append(prob)



    for count, doc_position in enumerate(sorted_sims, 1):
        score = similarities[doc_position]
        print(f"Document id: {doc_position}, name: {data['n'][doc_position]} with similarity score: {score:.4f}")
        rank.append(count)
        names.append(data['n'][doc_position])
        publication.append(data['paper_list'][doc_position])
        affiliation.append(data['affiliation'][doc_position])
        a = data['t'][doc_position].replace(";", " ")
        kw.append(a)
        sim.append(score)

    df1 = {
        'LDA_rank': rank,
        'LDA_Name': names,
        'Score': sim,
        'Keywords_LDA': kw,
        'Publication': publication,
        'Affiliation': affiliation
    }

    df2 = {
        'Topic': top,
        'Words': topic_words,
        'Probability': topic_prob
    }

    return df1, df2
def reset_version_state():
                # Clear any previous version state 
                st.session_state.prediction_ready = False
                st.session_state.initial_prompt_sent = False
                st.session_state.chat_history = []
                st.session_state.chat_html = ""
                st.session_state.explain_clicked = False
                st.session_state.show_explain_option = False
                st.session_state.question_asked = 0
                st.session_state.session_id=None 

def render_recommender_page(version, scenario_index):
    st.title("Grad Student Advisor Recommender System")
    st.divider()

    st.markdown("_Grad Student Scenario:_")

    scenario = st.session_state.selected_scenarios[scenario_index]

    start_session(
        st.session_state.user_name,
        " ".join(scenario.split()[:2]),
        version
    )

    st.info(scenario)

    st.markdown(
        "Enter keywords of your research interest separated by comma and get the system's recommendations."
    )

    keywords = st.multiselect(
        "Select Research Keywords:",
        options=options,
        key=f"keywords_{version}"
    )

    return scenario, keywords
                
def render_v1(scenario):
    explain_container = st.empty()
    student_name = " ".join(scenario.split(" ")[:2])
    advisor_name_1=st.session_state["cosine"]['Name'][0]
    advisor_name_2=st.session_state["lda1"]['LDA_Name'][0]

    questions = [
        "How does the system work?",
        f"How {advisor_name_1} and {advisor_name_2} are recommended?",
        f"What if {student_name} had different keywords, how would that change the results?",
    ]

    # if DEBUG:
    #     questions = ['Hi', 'Thank you','ok']
    if st.session_state.show_explain_option:
        # explain_container.markdown("**Do you want a more detailed explanation?**")
        if countdown_with_button("Please read the results carefully", st.session_state.get("COOLDOWN_TIME_SHORT", COOLDOWN_TIME_SHORT), questions[0], "explain_btn"):
            st.session_state.show_explain_option = False
            st.session_state.explain_clicked = True
            explain_container.empty()
            st.rerun()

    if st.session_state.explain_clicked:
        st.markdown("### 💬 Explanation and Follow-ups")

    chat_box = st.empty()
    # if st.session_state.initial_prompt_sent or st.session_state.explain_clicked:
    #     render_chat_styles(chat_box)

    if st.session_state.explain_clicked:
        chat_box = start_llm_chat(scenario, questions)

    if st.session_state.initial_prompt_sent:
        continue_llm_chat(questions)


def render_chat_transcript():
    """Render full transcript from session_state.chat_history (exclude system)."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            continue
        if msg["role"] == "dummy":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def start_llm_chat(scenario, questions):
    # Build system prompt & initialize history
    system_prompt = make_system_prompt(
        st.session_state.selected_keywords,
                    scenario,
        core_system_knowledge=CORE_SYSTEM_KNOWLEDGE
    )

    chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": questions[0]}
    ]
    st.session_state.chat_history=chat_history
    st.session_state.initial_prompt_sent = True
    st.session_state.explain_clicked = False
    st.session_state.followup_idx = 1  # reset index

    # Always render what's in history first
    render_chat_transcript()

    # Stream only for the active question (first one)
    with st.chat_message("assistant"):
        response_container = st.empty()
        assistant_text = ""
        print("start_llm_chat->stream_llm_api")
        for chunk in stream_llm_api(chat_history):
            assistant_text += chunk
            response_container.markdown(assistant_text + "▌")
        response_container.markdown(assistant_text)

    # Append the streamed message to history (so it shows next rerun)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
    log_chat_message("user", questions[0])
    log_chat_message("assistant", assistant_text)
    st.rerun()  # force rerun so transcript now includes it

def continue_llm_chat(questions):
    idx = st.session_state.followup_idx

    # Initialize streaming flag if not set
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False

    # Render the chat transcript so far
    render_chat_transcript()

    # If streaming is active, stream the assistant's reply now
    if st.session_state.is_streaming:
        with st.chat_message("assistant"):
            response_container = st.empty()
            assistant_text = ""
            print("start_llm_chat->stream_llm_api")
            for chunk in stream_llm_api(st.session_state.chat_history):
                assistant_text += chunk
                response_container.markdown(assistant_text + "▌")
            response_container.markdown(assistant_text)
        
        # Update streaming flag and rerun after done
        st.session_state.is_streaming = False
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
        st.rerun()

    # Only if not streaming, render buttons/forms for next user input
    else:
        def ask_and_advance(q):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state.is_streaming = True  # Set streaming flag to True for next rerun
            st.session_state.followup_idx += 1
            st.rerun()

        # Scripted questions buttons
        if idx < len(questions):
            if countdown_with_button(
                message="Please read the generated text carefully",
                duration_sec=st.session_state.get("COOLDOWN_TIME_LONG", COOLDOWN_TIME_LONG),
                button_label=questions[idx],
                button_key=f"followup_btn_{idx}"
            ):
                ask_and_advance(questions[idx])
        # Free-form input
        else:
            st.markdown("---")
            st.markdown("#### 📝 Do you have any other questions?")
            user_input = countdown_with_form(
                message="Please read carefully before interacting with the chatbot",
                duration_sec=st.session_state.get("COOLDOWN_TIME_LONG", COOLDOWN_TIME_LONG),
                form_key="freeform_followup",
                input_key="freeform_input"
            )
            if user_input:
                ask_and_advance(user_input)


        
def load_scenarios(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        scenarios = file.read().split("---")  # "---" as a separator
    scenarios = [ x.strip() for x in scenarios]
    return scenarios
# --- Session state inits ---
for key, default in [
    ("page","home"),
    ("user_name",""),
    ("prediction_ready", False),
    ("initial_prompt_sent", False),
    ("chat_history", []),
    ("chat_html", ""),
    ("explain_clicked", False),
    ("selected_symptoms_clean", None),
    ("show_explain_option",False),
    ("question_asked",0),
    ("questions",["How does the system work?","Tell me how the first advisor in text similarity model recommended?"]),
    ("scenarios_loaded",False)
]:
    if key not in st.session_state:
        st.session_state[key] = default


if 'clicked' not in st.session_state:
    st.session_state.clicked = False

questions=["How does the system work?","Tell me how the first advisor in text similarity model recommended?"]
data_dict={}
flag=0


if st.session_state.page == "home":
    st.title("Grad Student Advisor Recommender System")
    if st.session_state.user_name == "":
        st.markdown("### Please enter your name to get started:")

        # 4.1) Name input
        name_col, submit_col = st.columns([3, 1], vertical_alignment="bottom")
        with name_col:
            if st.session_state["user_name"] == "":
                st.session_state["user_name"] = st.text_input("Your name:", value="", placeholder="Type your name here")
                
        
        with submit_col:
            if st.button("Start"):
                if st.session_state["user_name"].strip() == "":
                    st.warning("Please enter at least one character for your name.")
                else:
                    st.success(f"Hello, {st.session_state['user_name'].strip()}!")
                    st.rerun()
    
    if not st.session_state.scenarios_loaded:            
        scenarios = load_scenarios("grad_student_scenario.md")
        random.shuffle(scenarios)
        st.session_state.selected_scenarios = scenarios
        st.session_state.scenarios_loaded = True
    # Only show the welcome text & cards if we have a name
    if st.session_state["user_name"].strip() != "":
        st.markdown(f"#### Hi, **{st.session_state['user_name']}**! Please choose a version below:")
        st.markdown("")

        # 4.2) Two card-style buttons
        c1, c2 = st.columns(2, gap="large")
          
        with c1:
            st.info("""
            **Version 1**: Standard Explanation + Follow-Up Questions.

            Enter your research keywords to receive advisor recommendations.

            You will receive a standard natural language explanation and may ask follow-up questions to the AI chatbot.
            """)

            if st.button("Go to Version 1"):
                st.session_state.page = "v1"
                reset_version_state()
                st.rerun()


        with c2:
            st.info("""
            **Version 2**: Standard Explanation + Quiz-Based Prompting.

            Enter your research keywords to receive advisor recommendations.

            You will receive a standard explanation plus 3 quiz-based prompts with explanatory feedback from the AI.
            """)

            if st.button("Go to Version 2"):
                st.session_state.page = "v2"
                reset_version_state()
                st.rerun()


        c3, c4 = st.columns(2, gap="large")

        with c3:
            st.info("""
            **Version 3**: Analogy-Based Explanation + Follow-Up Questions.

            Enter your research keywords to receive advisor recommendations.

            You will receive an analogy-based explanation and may ask follow-up questions to the AI chatbot.
            """)

            if st.button("Go to Version 3"):
                st.session_state.page = "v3"
                reset_version_state()
                st.rerun()


        with c4:
            st.info("""
            **Version 4**: Analogy-Based Explanation + Quiz-Based Prompting.

            Enter your research keywords to receive advisor recommendations.

            You will receive an analogy-based explanation plus 3 quiz-based prompts with analogy-based explanatory feedback from the AI.
            """)

            if st.button("Go to Version 4"):
                st.session_state.page = "v4"
                reset_version_state()
                st.rerun()
                
                

elif st.session_state.page == "v1" or st.session_state.page == "v2" or st.session_state.page == "v3" or st.session_state.page == "v4":
    # --- UI ---
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            reset_version_state()
            st.rerun()
    if st.session_state.page == "v1":
        scenario, keywords = render_recommender_page("v1", 0)

    if st.session_state.page == "v2":
        scenario, keywords = render_recommender_page("v2", 1)

    if st.session_state.page == "v3":
        scenario, keywords = render_recommender_page("v3", 2)

    if st.session_state.page == "v4":
        scenario, keywords = render_recommender_page("v4", 3)
    if st.button("Predict"):
        if len(keywords) < 1:
            st.warning("Please select at least one keyword.")
        else:
            reset_lock_timer()
            reset_prequiz_states()
            initialize_v2()
            import time
            name=st.session_state.user_name
            keywords = ", ".join(keywords[:-1]) + f", and {keywords[-1]}" if len(keywords) > 1 else keywords[0]
            st.session_state["selected_keywords"] = keywords
            with st.spinner(text="Hello "+name+"! Please wait while we retrieve some information."):
                output=cosine_recommender(keywords)           
                data_dict = json.loads(output)
                
                lda1,lda2=LDA(keywords)          
                st.session_state["cosine"] = data_dict
                with open('rec_result.txt', 'w') as f:
                            msg="User name is "+ name+". User reseach interests are "+keywords+". Top 3 recommended advisor list based on Cosine similarity:\n"
                            for i in range(len(data_dict['Ranking'])):
                                msg+=str(i+1)+'. name: '+ data_dict['Name'][i]
                                #if i==0:
                                #        st.session_state.questions.append(f"Why was Dr. {data_dict['Name'][i]} recommended?")
                                msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                                msg+='. Keywords: '+data_dict['Keywords'][i]+'\n'
                                msg+='. Publication: '+data_dict['Publication'][i]+'\n'
                                msg+='. Affiliaiton: '+data_dict['Affiliation'][i]+'\n'
                            f.write(msg)
                st.session_state["lda1"] = lda1
                with open('rec_result.txt', 'a') as f:
                            msg="\nTop 3 recommended advisor list based on LDA Topic modeling:\n"
                            for i in range(len(lda1['LDA_rank'])):
                                msg+=str(i+1)+'. name: '+ lda1['LDA_Name'][i]
                                msg+='. Similarity score: '+str(lda1['Score'][i])
                                msg+='. Keywords: '+lda1['Keywords_LDA'][i]+'\n'
                                msg+='. Publication: '+lda1['Publication'][i]+'\n'
                                msg+='. Affiliation: '+lda1['Affiliation'][i]+'\n'
                            f.write(msg)
                st.session_state["lda2"] = lda2
                with open('rec_result.txt', 'a') as f:
                            msg="\nTop LDA Topic selected:\n"
                            for i in range(len(lda2['Topic'])):
                                msg+=str(i+1)+'. Topic id: '+ str(lda2['Topic'][i])
                                #msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                                msg+='. Keywords: '+" ".join(lda2['Words'][i])+'\n'
                            f.write(msg)
                
                st.session_state.prediction_ready=True
                st.session_state.show_explain_option = True
                msg="" 
                with open('rec_result.txt', 'r') as f:
                                    for line in f:
                                        msg+=line
                                        
                prompt="""
                You are an AI-powered academic advisor chatbot designed to explain the reasoning behind advisor recommendations generated by a machine learning system. Your goal is to help prospective graduate students understand how their research interests align with those of faculty members based on two recommendation models.

Below is the **system design** as implemented:

1. **Text Similarity Model (Cosine Similarity):**
   - Inputs: Research keywords provided by the user.
   - Each advisor’s research profile is represented as a numerical count vector of publication keywords.
   - Cosine similarity is calculated between the user’s keyword vector and each advisor’s vector.
   - Output: Top 3 advisors with the highest similarity scores (range: 0 to 1), where values closer to 1 indicate stronger alignment.

2. **Topic Similarity Model (LDA Topic Modeling):**
   - Inputs: User’s research keywords mapped to 30 predefined LDA topics.
   - Each advisor has a topic distribution profile learned from their publication data.
   - The similarity between the user’s topic vector and each advisor’s topic profile is computed.
   - Output: Top 3 advisors with the most similar topic distributions.

**System Inputs and Outputs for This Session:**
     
                """
                prompt2="""**Expected Outcome:**
Help users interpret why these advisors were recommended, how closely their research interests align, and how changes in keywords might affect the results. Word counts more than 200.

**Guidelines:**
- Provide explanations in details why the advisors are recommended.
- You can answer both **general** and **scenario-specific** questions.

**For General Questions** (e.g., *"How does the system work?"*):
- Explain both models:
  • How keyword similarity (cosine similarity) works like comparing the direction of two arrows.
  • How LDA groups keywords into research themes and compares distributions.
- Explain why using both models gives a more robust match.
- Provide Feature-Based Explanation: Explains how users individual research keywords contributed to the results. Provide example using users individual research keywords contributes from rank 1 to rank 3 similar advisor.
- Counterfactual-Based Explanation: Shows how changing reseach keywords would alter predictions. Provide example using users individual research keywords change can make rank 3 to rank 1.
- Model Inner Working with Simple Example: Provides a basic calculation with example of how the system makes decisions. Provide example such as [kw1, kw2,..] to vector using users individual research keywords. Then a similarity score example using dot product. Also a LDA group of words.

**For Scenario-Specific Questions** (e.g., *"Why the top advisor recommended?"*):
- Explain how the user’s keywords closely matched the advisor’s keywords or topics.
- Show which terms contributed to high similarity. Show a dot product calculation.
- Mention concrete alignment in research themes.
- Highlight key differences between text vs. topic model rankings.
- Give “what-if” examples—how changing or refining keywords might change recommendations.
- Clarify what the similarity scores mean and that a lower score can still be meaningful in niche areas.

You are now ready to answer the user’s questions about their recommended graduate advisors.
                """
                st.session_state.messages = [{'role':'system', 'content':prompt+msg+prompt2}]
                    #response="Welcome "+name+"! Would you like an explanation of your recommendation for advisors?"
                #response = st.write_stream(chat_stream())

                #print(msg)
                #st.session_state.messages.append({"role": "assistant", "content": response})
                    #connection = connect_to_db()
                    #insert_message(connection, "LLM", response)
                    #connection.close()
                
                    
    if not st.session_state.prediction_ready:
                render_spacer()            
    if st.session_state.prediction_ready:
                df1 = pd.DataFrame(st.session_state["cosine"])
                df2 = pd.DataFrame(st.session_state["lda1"])
                df3 = pd.DataFrame(st.session_state["lda2"])
                
                #left_column, right_column = st.columns(2)
                #left_column, right_column = st.tabs(["Text Similarity", "Topic Similarity"])
                #with left_column:
                
                df1_new = df1[['Ranking','Name','Keywords', 'Similarity Score','Publication','Affiliation']]                
                df1_new = df1_new.to_dict(orient='records')
                st.write("Top 3 recommended advisor based on Text Similarity of keywords:")
                st.dataframe(df1_new, hide_index=True,  column_config={
                "Publication": st.column_config.Column(
                width="large",
                required=True,
                ),
                "Keywords": st.column_config.Column(
                width="medium",
                required=True,
                ),
                "Affiliation": st.column_config.Column(
                width="medium",
                required=True,
                )
                },)

                #with right_column:
                st.write("Top 3 recommended advisor based on LDA Topic Similarity of 30 topics:")
                df2_new = df2[['LDA_rank','LDA_Name','Keywords_LDA','Publication','Affiliation']] 
                df2_new = df2_new.to_dict(orient='records')
                st.dataframe(df2_new,hide_index=True, column_config={
                "LDA_rank": "Ranking","LDA_Name": "Name", "Publication": st.column_config.Column(
                width="large",
                required=True,
                ),
                "Keywords_LDA": st.column_config.Column(
                width="medium",
                required=True,
                ),
                })
                st.write("Double clicking individual cell will provide detail texts.")
                if st.session_state.page == "v3":
                        st.markdown("---")
                        
                        st.markdown("## 📋 How the Advisor Recommender System Works")
                        
                        st.markdown("""
                        Our system is designed to help prospective graduate students find suitable research advisors by matching them based on shared research interests and publications. The system uses two models: a **Text Similarity Model** and a **Topic Similarity Model**, each generating the top three advisor recommendations based on the user’s input keywords.
                        
                        The **Text Similarity Model** converts research keywords from publications into numerical count vectors and uses cosine similarity to measure how closely a user’s research interests align with those of potential advisors. A score closer to 1 indicates a stronger match, and the top three advisors with the highest similarity scores are recommended.
                        
                        The **Topic Similarity Model** employs Latent Dirichlet Allocation (LDA) to categorize publication keywords into 30 thematic clusters. Each advisor is assigned probability scores across these topics, creating a thematic profile. The system matches the user’s input keywords to these topics and recommends the top three advisors whose profiles align most closely with the user’s interests.
                        
                        Results are displayed in two tabs: one for Text Similarity and one for Topic Similarity, each showing advisors’ names, affiliations, and publication details. The recommendations aim to foster meaningful academic collaborations by aligning students with advisors whose research interests are most compatible.
                        """)
                        
                        with st.expander("**Key Terms**", expanded=True):
                            st.markdown("""
                        - **Text Similarity Model:** Converts publication keywords into count vectors and uses cosine similarity to measure alignment with user interests.
                        - **Topic Similarity Model:** Uses Latent Dirichlet Allocation (LDA) to group keywords into 30 topics and matches user interests to advisors’ thematic profiles.
                        - **Cosine Similarity:** A score (0 to 1) indicating how closely two sets of keywords align; higher scores mean greater similarity.
                        - **Latent Dirichlet Allocation (LDA):** A model that groups keywords into thematic clusters to identify research topics.
                        - **Count Vector:** A numerical representation of keywords, where each value indicates the presence or frequency of a keyword.
                        """)
                        # Link to Quiz
                        st.markdown("---")
                        
                        reveal_button_html = """
                                <a href="https://quiz-rec.streamlit.app/" target="_blank">
                                        <button style="
                                            background-color:#4CAF50;
                                            border:none;
                                            color:white;
                                            padding:10px 20px;
                                            text-align:center;
                                            text-decoration:none;
                                            display:inline-block;
                                            font-size:16px;
                                            border-radius:5px;
                                            cursor:pointer;">
                                            Go to Quiz
                                        </button>
                                    </a>
                                """
                        countdown_component_html("Please read the given text carefully", COOLDOWN_TIME_LONG, reveal_button_html)

                if st.session_state.page == "v2":
                        render_v2(st.session_state.selected_scenarios[1])
                       
                if st.session_state.page == "v1":
                        render_v1(st.session_state.selected_scenarios[0])
                        
                        # if "messages"  in st.session_state:
                        #     for message in st.session_state.messages:
                        #         if message['role']=='system':
                        #             continue
                        #         if message['role']=='user':
                        #             with st.chat_message(message["role"],avatar="👦"):
                        #                 st.markdown(message["content"])
                        #         else:
                        #             with st.chat_message(message["role"]):
                        #                 st.markdown(message["content"])
                        #     def ask_and_advance(i):
                        #             st.session_state.messages.append({"role": "user", "content": st.session_state.questions[i]})                                 
                        #             response = st.write_stream(chat_stream())                               
                        #             st.session_state.question_asked+=1                                  
                        #             st.session_state.messages.append({"role": "assistant", "content": response})
                        #     if st.session_state.question_asked<2:
                        #                         if countdown_with_button(
                        #                                     message="Please read the generated text carefully",
                        #                                     duration_sec=COOLDOWN_TIME_SHORT,
                        #                                     button_label=questions[st.session_state.question_asked],
                        #                                     button_key=f"followup_btn_{st.session_state.question_asked}"
                        #                                 ):
                        #                                 ask_and_advance(st.session_state.question_asked)
                        #                                 st.rerun()
                                                       
                                                        
                                                
                        #     if st.session_state.question_asked>=2:
                        #             prompt=countdown_with_form(
                        #                             message="Please read carefully before interacting with the chatbot",
                        #                             duration_sec=COOLDOWN_TIME_LONG,
                        #                             form_key="freeform_followup",
                        #                             input_key="freeform_input"
                        #                         )
                        #             if prompt : 
                        #                 #st.chat_input("Example: 1. Tell me the research interests of the recommended advisor based on cosine similarity. \n2. Tell me why 'X' is recommended.\n 3. What is cosine similarity."):
                        #                 st.session_state.messages.append({"role": "user", "content": prompt})
                        #                 with st.chat_message("user",avatar="👦"):
                        #                     st.markdown(prompt)
                
                        #                 with st.chat_message("assistant"):    
                        #                     response = st.write_stream(chat_stream())
                        #                 st.session_state.messages.append({"role": "assistant", "content": response})
                        #                 st.rerun()
                
                                                
                
