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
import re
from pathlib import Path

st.set_page_config("Advisor Recommendation", page_icon=":book:",   layout="wide")

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
    username=st.session_state["user_name"]
    version=st.session_state["page"]

    payload = {
        "session_id": sid,
        "role": role,                   # 'user' | 'assistant' | 'system'
        "content": content,
        "username": username,
        "version": version
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




import html
import uuid
import pandas as pd
import streamlit as st


def format_table_value(value, col_name):
    """Format values before showing them in the recommendation table."""
    if pd.isna(value):
        return ""

    if col_name == "Ranking":
        try:
            return str(int(value))
        except Exception:
            return str(value)

    if col_name == "Similarity Score":
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    return str(value)


def render_wrapped_recommendation_table(
    df,
    title,
    columns,
    rename=None,
    col_widths=None,
    max_height=520
):
    """
    Display a readable recommendation table with wrapped text.
    This removes horizontal scrolling and lets users read long cells vertically.
    """

    table_df = df.loc[:, columns].rename(columns=rename or {}).copy()
    table_df = table_df.fillna("")

    if col_widths is None:
        col_widths = [100 / len(table_df.columns)] * len(table_df.columns)

    if len(col_widths) != len(table_df.columns):
        raise ValueError("col_widths must match the number of displayed columns.")

    table_id = f"rec_table_{uuid.uuid4().hex}"

    colgroup_html = "".join(
        f"<col style='width: {width}%'>"
        for width in col_widths
    )

    header_html = "".join(
        f"<th>{html.escape(str(col))}</th>"
        for col in table_df.columns
    )

    rows_html = []

    for _, row in table_df.iterrows():
        cell_html = []

        for col in table_df.columns:
            value = format_table_value(row[col], col)
            css_class = "num-cell" if col in ["Ranking", "Similarity Score"] else ""

            cell_html.append(
                f"<td class='{css_class}'>{html.escape(value)}</td>"
            )

        rows_html.append(f"<tr>{''.join(cell_html)}</tr>")

    st.write(f"**{title}**")

    st.markdown(
        f"""
        <style>
            #{table_id}_wrapper {{
                width: 100%;
                max-height: {max_height}px;
                overflow-y: auto;
                overflow-x: hidden;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 0.35rem;
                margin-bottom: 1.25rem;
            }}

            #{table_id} {{
                width: 100%;
                table-layout: fixed;
                border-collapse: collapse;
                font-size: 0.92rem;
            }}

            #{table_id} th,
            #{table_id} td {{
                border-bottom: 1px solid #e5e7eb;
                border-right: 1px solid #e5e7eb;
                padding: 10px 12px;
                text-align: left;
                vertical-align: top;
                white-space: normal;
                overflow-wrap: anywhere;
                word-break: break-word;
                line-height: 1.35;
            }}

            #{table_id} th {{
                position: sticky;
                top: 0;
                background-color: #f8fafc;
                z-index: 1;
                font-weight: 600;
            }}

            #{table_id} tr:last-child td {{
                border-bottom: none;
            }}

            #{table_id} th:last-child,
            #{table_id} td:last-child {{
                border-right: none;
            }}

            #{table_id} .num-cell {{
                text-align: right;
                white-space: nowrap;
            }}
        </style>

        <div id="{table_id}_wrapper">
            <table id="{table_id}">
                <colgroup>
                    {colgroup_html}
                </colgroup>
                <thead>
                    <tr>{header_html}</tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    
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
                        Our system is designed to help prospective graduate students find suitable research advisors by matching them based on shared research interests and publications. On the recommendation page, users select research keywords from a predefined keyword list (e.g., "software quality", "software system", "case studies") and click the Predict button. The system then runs two models and displays both sets of results on the same page: a **Text Similarity Model** and a **Topic Similarity Model**, each generating the top three advisor recommendations.

                        1. **Text Similarity Model (Cosine Similarity):**
                           - Inputs: Research keywords selected by the user.
                           - Each advisor's research profile is represented as a numerical count vector of publication keywords.
                           - Cosine similarity is calculated between the user's keyword vector and each advisor's vector.
                           - Output: A table titled "Top 3 recommended advisors based on Text Similarity of keywords" showing each advisor's Name, Keywords, Text Similarity score, Publications, and Affiliation.
                           - Scores range from 0 to 1 (shown to four decimal places); values closer to 1 indicate stronger alignment with the user's exact keywords.

                        2. **Topic Similarity Model (LDA + Cosine Similarity Hybrid):**
                           - Inputs: User's selected research keywords, mapped against 30 topics learned by an LDA topic model.
                           - Step 1: The single most relevant topic (highest probability) is selected for the user's keywords. The page displays the selected topic number and its top 10 topic words (e.g., "Selected Topic: 19. Keywords for selected topic: softwar, video, develop, ...").
                           - Step 2: These topic words are used as an expanded query, and cosine similarity is computed between this topic-word query and each advisor's publication keyword count vector (same method as the Text Similarity Model).
                           - Output: A table titled "Top 3 recommended advisors based on selected LDA Topic" showing Name, Keywords (LDA), Topic Similarity score, Publications, and Affiliation.
                           - This approach recommends advisors based on the broader research theme of the user's keywords rather than the exact keywords, so it can surface advisors in the same topic area even if their terminology differs. Topic Similarity scores are therefore often lower than Text Similarity scores, since the query is the topic's words rather than the user's exact keywords.

                        **Important display notes:**
                        - Topic words are stemmed model terms, so some words look shortened (e.g., "softwar" for "software", "qualiti" for "quality", "engin" for "engineering"). These are not spelling errors.
                        - Each advisor entry lists their research keywords, up to five representative publications, and one or more affiliations (advisors may be associated with multiple institutions).
                        - The two result tables may recommend different advisors: the Text Similarity table reflects exact keyword overlap, while the Topic Similarity table reflects thematic overlap.
                        """

def make_system_prompt(
    selected_topic,
    scenario,
    version="v1",
    core_system_knowledge=CORE_SYSTEM_KNOWLEDGE
):
    """
    Construct the system prompt for advisor recommendation explanation.

    version == "v1": Standard explanation with follow-up questions
    version == "v3": Analogy-based explanation with follow-up questions
    """
    version=st.session_state.page
    data_dict = st.session_state["cosine"]
    lda1 = st.session_state["lda1"]
    lda2 = st.session_state["lda2"]

    # User educational background for analogy-based explanation
    field_of_study = st.session_state.get("field_of_study", "Not provided")
    specific_topics = st.session_state.get("specific_topics", "Not provided")
    background_keywords = st.session_state.get("background_keywords", "Not provided")

    recommendation_context = f"""
    ## Current Student Context

    - Student Scenario: {scenario}
    - User selected topics: {selected_topic}

    - Top 3 recommended advisor list based on Cosine similarity:
        1. Name: {data_dict['Name'][0]}; Cosine similarity score: {data_dict['Similarity Score'][0]}; Keywords: {data_dict['Keywords'][0]}; Publication: {data_dict['Publication'][0]}; Affiliation: {data_dict['Affiliation'][0]}
        2. Name: {data_dict['Name'][1]}; Cosine similarity score: {data_dict['Similarity Score'][1]}; Keywords: {data_dict['Keywords'][1]}; Publication: {data_dict['Publication'][1]}; Affiliation: {data_dict['Affiliation'][1]}
        3. Name: {data_dict['Name'][2]}; Cosine similarity score: {data_dict['Similarity Score'][2]}; Keywords: {data_dict['Keywords'][2]}; Publication: {data_dict['Publication'][2]}; Affiliation: {data_dict['Affiliation'][2]}

    - Top 3 recommended advisor list based on LDA Topic modeling:
        1. Name: {lda1['LDA_Name'][0]}; LDA similarity/topic score: {lda1['Score'][0]}; Keywords: {lda1['Keywords_LDA'][0]}; Publication: {lda1['Publication'][0]}; Affiliation: {lda1['Affiliation'][0]}
        2. Name: {lda1['LDA_Name'][1]}; LDA similarity/topic score: {lda1['Score'][1]}; Keywords: {lda1['Keywords_LDA'][1]}; Publication: {lda1['Publication'][1]}; Affiliation: {lda1['Affiliation'][1]}
        3. Name: {lda1['LDA_Name'][2]}; LDA similarity/topic score: {lda1['Score'][2]}; Keywords: {lda1['Keywords_LDA'][2]}; Publication: {lda1['Publication'][2]}; Affiliation: {lda1['Affiliation'][2]}

    - Top LDA Topic selected:
        Topic id: {lda2['Topic'][0]}
        Keywords: {lda2['Words'][0]}
    """

    # -----------------------------
    # Version 1: Standard explanation
    # -----------------------------
    if version == "v1" or version == "v2" :
        prompt = f"""
        You are an AI-powered academic advisor chatbot designed to explain the reasoning behind advisor recommendations generated by a machine learning system.

        Your goal is to help prospective graduate students understand how their research interests align with faculty members based on two recommendation models.

        ---
        ## System Knowledge

        {core_system_knowledge}

        ---
        ## System Inputs and Outputs for This Session

        {recommendation_context}

        ---
        ## Expected Outcome

        Help users interpret:
        - why these advisors were recommended,
        - how closely their research interests align,
        - how the two recommendation models work,
        - and how changes in keywords might affect the results.

        Responses should usually be more than 200 words when the user asks for an explanation.

        ---
        ## Guidelines

        - Provide detailed explanations of why the advisors are recommended.
        - You can answer both general and scenario-specific questions.
        - Keep the explanation grounded in the provided recommendation results.
        - Do not invent advisor information beyond the provided context.
        - Use clear language and avoid unnecessary technical jargon.

        ---
        ## For General Questions

        Example user questions:
        - "How does the system work?"
        - "How are advisors recommended?"
        - "What is cosine similarity?"
        - "What is LDA?"

        You should:
        - Briefly explain how advisors are recommended.
        - Explain how keyword similarity, or cosine similarity, works.
        - Explain how LDA groups keywords into broader research themes.
        - Explain how the system compares the student profile with advisor profiles.
        - Explain why using both models gives a more robust recommendation.

        ---
        ## For Scenario-Specific Questions

        Example user questions:
        - "Why was this advisor recommended?"
        - "How were the top advisors from both models selected?"
        - "Why are the cosine and LDA rankings different?"

        You should:
        - Explain which student keywords contributed to high similarity.
        - Mention concrete keyword or topic alignment.
        - Show how the user's keywords matched advisor keywords or topic themes.
        - Explain the top LDA topic and how advisors were selected from that topic.
        - Highlight key differences between text-based ranking and topic-based ranking.
        - Clarify what similarity scores mean.
        - Explain that a lower score can still be meaningful in niche research areas.

        When useful, provide a simple vector example using the user's selected research keywords.

        Example:
        - User keywords: [machine learning, NLP, recommender systems, visualization]
        - Advisor keyword match vector: [1, 1, 1, 0]
        - Another advisor vector: [1, 0, 1, 0]

        Then briefly explain:
        - matching terms increase the dot product,
        - stronger overlap usually increases cosine similarity,
        - and higher similarity can lead to a higher advisor ranking.

        ---
        ## For What-If or Result Explanation Questions

        Example user questions:
        - "What if I selected different keywords?"
        - "Can you explain the results?"
        - "How could the third advisor become first?"

        You should provide:

        1. Feature-Based Explanation:
        - Explain how the user's individual research keywords contributed to the results.
        - Show how specific keywords helped rank one advisor above another.
        - Use the current selected keywords when possible.

        2. Counterfactual-Based Explanation:
        - Explain how changing, adding, or removing keywords could alter the rankings.
        - Give an example of how a rank 3 advisor could move to rank 1 if the user's keywords matched that advisor more strongly.

        You are now ready to answer the user's questions about their recommended graduate advisors.
        """

    # --------------------------------
    # Version 3: Analogy-based explanation
    # --------------------------------
    elif version == "v3" or version == "v4" :
        prompt = f"""
        You are an AI explanation assistant for a Grad Student Advisor recommender system.

        Your task is to explain how the advisor recommender system generated recommendations using an analogy based on the participant's educational background and detailed topic interest.

        ---
        ## System Information

        {core_system_knowledge}

        ---
        ## Participant Educational Background

        - Current or most recent field of study: {field_of_study}
        - Detailed topic interest within that background: {specific_topics}
        - Familiar concepts or keywords from that topic: {background_keywords}

        ---
        ## System Inputs and Outputs for This Session

        {recommendation_context}

        ---
        ## Main Goal

        Help users understand the advisor recommendation process through an analogy grounded in their educational background.

        The explanation should help users understand:
        - how the system interprets the student's research interests,
        - how the system compares the student's profile with advisor profiles,
        - how cosine similarity and LDA topic modeling work,
        - how similarity scores or topic matches affect rankings,
        - and why the final advisors were recommended.

        Responses should usually be more than 200 words when the user asks for an explanation.

        ---
        ## Analogy-Based Explanation Requirements

        When answering the user, generate an explanation that satisfies the following requirements:

        1. Explain how the system interprets the student's research interests.
        2. Explain how the system compares the student's profile with advisor profiles.
        3. Explain how text similarity, topic similarity, similarity scores, or topic matches are used to identify stronger matches.
        4. Explain how the system ranks advisors and produces the final recommendation.
        5. Use an analogy grounded in the participant's educational background and detailed topic interest.
        6. Clearly map each part of the analogy to the recommender system process.
        7. Preserve the technical meaning of the original recommendation process.
        8. Avoid misleading, inaccurate, or overly complex mappings.
        9. Use clear, concise, and non-technical language.

        ---
        ## Required Analogy Mapping

        Whenever you use an analogy, clearly connect the analogy parts to the recommender system parts.

        For example, explain mappings like:

        - The student's research keywords are like important features, concepts, materials, cases, themes, or design elements in the participant's field.
        - Advisor profiles are like possible matches, categories, examples, audiences, agencies, artworks, policies, or cases.
        - Cosine similarity is like measuring how much the student's selected features overlap with each advisor's features.
        - LDA topic modeling is like grouping detailed keywords into broader themes.
        - Ranking advisors is like selecting the best-fitting option based on the strongest overall match.

        Use the participant's actual field, topic, and keywords whenever possible.

        ---
        ## For General Questions

        Example user questions:
        - "How does the system work?"
        - "How are advisors recommended?"
        - "What is cosine similarity?"
        - "What is LDA?"

        You should:
        - Explain how advisors are recommended using an analogy from the user's educational background.
        - Explain cosine similarity through that analogy.
        - Explain LDA topic modeling through that analogy.
        - Explain why using both models gives a more complete recommendation.
        - Keep the analogy accurate and easy to follow.

        Example structure:
        - First explain the recommender system in simple terms.
        - Then introduce the analogy.
        - Then map the analogy back to the system.

        ---
        ## For Scenario-Specific Questions

        Example user questions:
        - "Why was this advisor recommended?"
        - "How were the top advisors from both models selected?"
        - "Why are the cosine and LDA rankings different?"

        You should:
        - Explain which student keywords contributed to high similarity.
        - Use the analogy to explain keyword overlap.
        - Explain the top LDA topic using the analogy.
        - Show how the user's keywords matched advisor keywords or topic themes.
        - Mention concrete alignment in research themes.
        - Highlight key differences between text-based ranking and topic-based ranking.
        - Clarify what similarity scores mean.
        - Explain that a lower score can still be meaningful in niche research areas.

        When useful, provide a simple analogy-based vector example.

        Example:
        - User selected concepts: [concept 1, concept 2, concept 3, concept 4]
        - Advisor A match pattern: [1, 1, 1, 0]
        - Advisor B match pattern: [1, 0, 1, 0]

        Then explain:
        - 1 means the advisor shares that concept,
        - 0 means the concept is not strongly present,
        - more meaningful overlap usually increases similarity,
        - and higher similarity can lead to a higher ranking.

        ---
        ## For What-If or Result Explanation Questions

        Example user questions:
        - "What if I selected different keywords?"
        - "Can you explain the results?"
        - "How could the third advisor become first?"

        You should provide:

        1. Feature-Based Explanation:
        - Explain how the user's individual research keywords contributed to the results.
        - Use the participant's educational analogy to explain why certain features increased the match.
        - Show how specific keywords helped rank one advisor above another.

        2. Counterfactual-Based Explanation:
        - Explain how changing, adding, or removing keywords could alter the rankings.
        - Use the analogy to explain how changing the input features changes the final match.
        - Give an example of how a rank 3 advisor could move to rank 1 if the user's keywords matched that advisor more strongly.

        ---
        ## Style Rules

        - Use the participant's actual field, topic, and keywords.
        - Do not use a generic analogy if participant background information is available.
        - Do not overextend the analogy.
        - Do not make misleading comparisons.
        - Preserve the correct technical meaning of cosine similarity, LDA, scores, keyword matches, and rankings.
        - Keep explanations supportive, educational, and easy to understand.
        - Avoid unnecessary jargon.
        - Do not invent advisor information beyond the provided context.

        You are now ready to answer the user's questions about their recommended graduate advisors using analogy-based explanations.
        """

    else:
        raise ValueError("version must be either 'v1' or 'v3'")

    return prompt.strip()

def clean_text(x):
    """Remove unwanted line breaks/extra spaces but keep the text content."""
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def render_rec_table(df):
    return df.to_html(
        index=False,
        escape=True,
        classes="rec-table"
    )

def make_quiz_system_prompt(
    question,
    options,
    correct_index,
    selected_topic,
    scenario,
    version="v2",
    core_system_knowledge=CORE_SYSTEM_KNOWLEDGE
):
    """
    Creates the system prompt for quiz-based explanation.

    version == "v2": Standard quiz-based explanation
    version == "v4": Analogy-based quiz explanation using user's educational background
    """
    version=st.session_state.page

    formatted_options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

    data_dict = st.session_state["cosine"]
    lda1 = st.session_state["lda1"]
    lda2 = st.session_state["lda2"]

    # User educational background for analogy-based explanation
    field_of_study = st.session_state.get("field_of_study", "Not provided")
    specific_topics = st.session_state.get("specific_topics", "Not provided")
    background_keywords = st.session_state.get("background_keywords", "Not provided")

    recommendation_context = f"""
        ## Current Student Context
        - Student Scenario: {scenario}
        - User selected topics: {selected_topic}

        - Top 3 recommended advisor list based on Cosine similarity:
            1. Name: {data_dict['Name'][0]}; Cosine similarity score: {data_dict['Similarity Score'][0]}; Keywords: {data_dict['Keywords'][0]}; Publication: {data_dict['Publication'][0]}; Affiliation: {data_dict['Affiliation'][0]}
            2. Name: {data_dict['Name'][1]}; Cosine similarity score: {data_dict['Similarity Score'][1]}; Keywords: {data_dict['Keywords'][1]}; Publication: {data_dict['Publication'][1]}; Affiliation: {data_dict['Affiliation'][1]}
            3. Name: {data_dict['Name'][2]}; Cosine similarity score: {data_dict['Similarity Score'][2]}; Keywords: {data_dict['Keywords'][2]}; Publication: {data_dict['Publication'][2]}; Affiliation: {data_dict['Affiliation'][2]}

        - Top 3 recommended advisor list based on LDA Topic modeling:
            1. Name: {lda1['LDA_Name'][0]}; LDA similarity/topic score: {lda1['Score'][0]}; Keywords: {lda1['Keywords_LDA'][0]}; Publication: {lda1['Publication'][0]}; Affiliation: {lda1['Affiliation'][0]}
            2. Name: {lda1['LDA_Name'][1]}; LDA similarity/topic score: {lda1['Score'][1]}; Keywords: {lda1['Keywords_LDA'][1]}; Publication: {lda1['Publication'][1]}; Affiliation: {lda1['Affiliation'][1]}
            3. Name: {lda1['LDA_Name'][2]}; LDA similarity/topic score: {lda1['Score'][2]}; Keywords: {lda1['Keywords_LDA'][2]}; Publication: {lda1['Publication'][2]}; Affiliation: {lda1['Affiliation'][2]}

        - Top LDA Topic selected:
            Topic id: {lda2['Topic'][0]}
            Keywords: {lda2['Words'][0]}
    """

    quiz_context = f"""
        ## Current Quiz Task
        The user is working through a pre-quiz designed to prepare them for a longer comprehension test.
        They are answering the following question:

        "{question}"

        Options:
        {formatted_options}

        The correct answer is option {correct_index}.
        The user will select one of the options and you will provide feedback based on their selection.
    """

    # -----------------------------
    # Version 2: Standard explanation
    # -----------------------------
    if version == "v2":
        prompt = f"""
        You are acting as an explanation assistant for a Grad Student Advisor recommender system.
        You have full internal knowledge of how the system works.

        ---
        ## System Knowledge
        {core_system_knowledge}

        ---
        {recommendation_context}

        ---
        {quiz_context}

        ---
        ## Special Instructions
        - If the user says "Option [OPTION NUMBER] has been selected", respond as follows:
            - If the user selects the correct answer, explain why it is correct. If not done yet, briefly explain how the system works.
            - If the user selects an incorrect answer, explain why it is wrong and guide them toward the correct reasoning.
        - The first time the user selects an option, give a brief explanation of the system and how it works, then answer based on the selected option.

        ## Your Role & Style Guide
        - Your main goal is to help the user understand the system reasoning and explain why the selected option is correct or incorrect.
        - Encourage step-by-step reasoning based on the system's recommendations, similarity scores, and reasoning logic.
        - Avoid generic advice; always tie reasoning back to how this specific system would think.
        - Keep explanations short, targeted, and context-aware.
        - Avoid long lectures.
        - If the user asks follow-up questions and seems unsure, ask small guiding questions rather than giving away the answer if they have not selected the correct option yet.
        - Use simple language and avoid technical jargon unless the user asks for it.

        Respond in a supportive and educational tone.
        """

    # --------------------------------
    # Version 4: Analogy-based explanation
    # --------------------------------
    elif version == "v4":
        prompt = f"""
        You are an AI explanation assistant for a Grad Student Advisor recommender system.
        Your task is to explain how the advisor recommender system generated a recommendation using an analogy based on the participant's educational background and detailed topic interest.

        ---
        ## System Information
        {core_system_knowledge}

        ---
        ## Participant Educational Background
        - Current or most recent field of study: {field_of_study}
        - Detailed topic interest within that background: {specific_topics}
        - Familiar concepts or keywords from that topic: {background_keywords}

        ---
        {recommendation_context}

        ---
        {quiz_context}

        ---
        ## Analogy-Based Explanation Requirements
        Generate feedback that satisfies the following requirements:

        1. Explain how the system interprets the student's research interests.
        2. Explain how the system compares the student's profile with advisor profiles.
        3. Explain how text similarity, topic similarity, similarity scores, or topic matches are used to identify stronger matches.
        4. Explain how the system ranks advisors and produces the final recommendation.
        5. Use an analogy grounded in the participant's educational background and detailed topic interest.
        6. Clearly map each part of the analogy to the recommender system process.
        7. Preserve the technical meaning of the original recommendation process.
        8. Avoid misleading, inaccurate, or overly complex mappings.
        9. Use clear, concise, and non-technical language.

        ---
        ## Special Instructions
        - If the user says "Option [OPTION NUMBER] has been selected", respond as follows:
            - If the user selects the correct answer, explain why it is correct using the educational-background analogy.
            - If the user selects an incorrect answer, explain why it is wrong and guide them toward the correct reasoning using the analogy.
        - The first time the user selects an option, briefly explain how the recommender system works using the analogy, then answer based on the selected option.
        - Do not create a vague analogy. Use the participant's actual field, topic, and keywords.
        - Always connect the analogy back to the advisor recommender system.
        - Keep the explanation short and focused.
        - Do not over-explain the analogy.
        - Do not change the correct answer.
        - Do not invent recommendation data beyond the provided context.

        ## Your Role & Style Guide
        - Your main goal is to help the user understand the system reasoning through a familiar educational analogy.
        - Tie the explanation to the selected option, advisor ranking, similarity scores, keyword matches, and topic matches.
        - Use simple language.
        - Avoid technical jargon unless the user asks for it.
        - Respond in a supportive and educational tone.
        """

    else:
        raise ValueError("version must be either 'v2' or 'v4'")

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
            log_chat_message("system", str(question))
            log_chat_message("user", selection_msg)

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
                    log_chat_message("user", user_input)
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
                    log_chat_message("assistant", assistant_text)
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
                        log_chat_message("user", user_input)
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
    rank, names, sim, kw, publication, affiliation = [], [], [], [], [], []
    top, topic_words, topic_prob = [], [], []

    # ---- Step 1: Preprocess user keywords ----
    new_doc = porter_stemmer(tokenize(keywords))
    new_doc_text = " ".join(new_doc)
    new_doc_vector = vectorizer.transform([new_doc_text])

    # ---- Step 2: Get the best (top-1) topic ----
    topic_distribution = lda_model.transform(new_doc_vector)[0]
    best_topic = topic_distribution.argmax()
    prob = topic_distribution[best_topic]
    print(f"Best Topic {best_topic} with probability {prob:.4f}")

    # ---- Step 3: Get top words of the best topic ----
    topic_terms = lda_model.components_[best_topic]
    top_words_idx = topic_terms.argsort()[-10:][::-1]
    words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    print(f"Top words for topic {best_topic}: {', '.join(words)}")

    top.append(best_topic)
    topic_words.append(words)
    topic_prob.append(prob)

    # ---- Step 4: Use topic words as a query document for cosine similarity ----
    topic_doc = " ".join(words)
    user_score, rec_adv = top_similar_doc_cosine(count_vector, topic_doc, 3)

    # ---- Step 5: Build top-3 advisor results (same as cosine_recommender) ----
    count = 1
    for i in rec_adv:
        rank.append(count)
        names.append(i)
        sim.append(user_score[i])
        publication.append(data[data['n'] == i]['paper_list'].values[0])
        affiliation.append(data[data['n'] == i]['affiliation'].values[0])

        a = data[data['n'] == i]['t']
        k = ""
        for j in a:
            k = " ".join(tokenize(j))
        kw.append(k)

        print(f"Rank {count}: {i} with similarity score {user_score[i]:.4f}")
        count += 1

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
        if countdown_with_button("Please read the results carefully", st.session_state.get("COOLDOWN_TIME_SHORT", COOLDOWN_TIME_SHORT), "Natural Language Explanation of given recommendation", "explain_btn"):
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
        {"role": "system", "content": system_prompt}
    ]
    st.session_state.chat_history=chat_history
    st.session_state.initial_prompt_sent = True
    st.session_state.explain_clicked = False
    st.session_state.followup_idx = 4  # reset index

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
    #log_chat_message("user", questions[0])
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
        log_chat_message("assistant", assistant_text)
        st.rerun()

    # Only if not streaming, render buttons/forms for next user input
    else:
        def ask_and_advance(q):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": q})
            log_chat_message("user", q)
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
# Initialize session state variables
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "field_of_study" not in st.session_state:
    st.session_state.field_of_study = ""

if "specific_topics" not in st.session_state:
    st.session_state.specific_topics = ""

if "background_keywords" not in st.session_state:
    st.session_state.background_keywords = ""
field_options = [
        "",
        "Communication",
        "Journalism and Media Communication",
        "Media Studies",
        "Public Relations and Advertising",
        "Fine Arts",
        "Studio Art",
        "Art History",
        "Music",
        "Theatre",
        "Creative Writing",
        "Public Administration",
        "Political Science",
        "Criminology and Criminal Justice",
        "Social Work",
        "Emergency Management",
        "Aviation",
        "Urban Studies",
        "Gerontology",
        "Other"
    ]
topic_examples = (
        "Examples: visual storytelling, media production, audience analysis, "
        "public policy, nonprofit management, criminal justice, social work, "
        "community engagement, public safety, art design, performance, communication strategy"
    )

keyword_examples = (
        "Examples: audience, message, framing, campaign, policy, governance, "
        "community, justice, service, creativity, design, performance, ethics"
    )


JSON_FILE = (
    Path(__file__).resolve().parent
    / "program_topics_with_descriptions.json"
)


@st.cache_data
def load_program_topic_data(file_path: str) -> dict:
    """Load and return the program-topic JSON data."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    program_topic_data = load_program_topic_data(str(JSON_FILE))
except FileNotFoundError:
    st.error(
        f"JSON file was not found:\n\n{JSON_FILE}\n\n"
        "Place the JSON file in the same folder as this Streamlit script."
    )
    st.stop()
except json.JSONDecodeError as error:
    st.error(f"The JSON file is not valid: {error}")
    st.stop()


# ============================================================
# Initialize session-state variables
# ============================================================

SESSION_DEFAULTS = {
    "page": "home",
    "user_name": "",
    "field_of_study": "",

    # Complete structured background information
    "background_profile": {},

    # Cosine-selection values
    "cosine_topic": "",
    "cosine_subtopics": [],
    "cosine_description": "",
    "cosine_analogy": "",

    # LDA-selection values
    "lda_topic": "",
    "lda_subtopics": [],
    "lda_description": "",
    "lda_analogy": "",

    # Existing variables retained for backward compatibility
    "specific_topics": "",
    "background_keywords": "",
}


for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# Helper functions
# ============================================================

def parse_custom_items(raw_text: str) -> list[str]:
    """
    Convert comma-, semicolon-, or line-separated text into a list.

    Duplicate entries are removed while preserving their original order.
    """
    items = [
        item.strip()
        for item in re.split(r"[,;\n]+", raw_text or "")
        if item.strip()
    ]

    return list(dict.fromkeys(items))


def empty_topic_selection() -> dict:
    """Return an empty model-topic selection."""
    return {
        "topic": "",
        "subtopics": [],
        "description": "",
        "analogy": "",
        "source": "",
    }


def render_topic_selection(
    method_key: str,
    topic_number: int,
    program_data: dict,
    program_widget_key: str,
) -> dict:
    """
    Render one topic-selection section.

    method_key is used internally to access either "cosine" or "LDA"
    from the JSON file. These technical terms are not shown to users.
    """

    st.markdown(f"#### Familiar Topic {topic_number}")

    available_topics = program_data.get(method_key, {})
    topic_names = list(available_topics.keys())

    # --------------------------------------------------------
    # Select a predefined or custom topic
    # --------------------------------------------------------

    if topic_names:
        selected_topic_option = st.selectbox(
            f"Select a topic you are familiar with for Topic {topic_number}:",
            options=[""] + topic_names + ["Other"],
            format_func=lambda value: (
                "Choose an option" if value == "" else value
            ),
            key=f"{method_key}_topic_{program_widget_key}",
        )
    else:
        selected_topic_option = "Other"

        st.caption(
            "No predefined topics are available for this program. "
            "Please enter a topic you are familiar with."
        )

    if selected_topic_option == "":
        return empty_topic_selection()

    # --------------------------------------------------------
    # Custom topic
    # --------------------------------------------------------

    if selected_topic_option == "Other":
        custom_topic = st.text_input(
            f"Enter a topic for Topic {topic_number}:",
            placeholder="Example: Educational leadership",
            key=f"{method_key}_custom_topic_{program_widget_key}",
        ).strip()

        custom_description = st.text_area(
            "Briefly describe this topic:",
            placeholder=(
                "Describe the main ideas covered by this topic in a few "
                "sentences."
            ),
            height=90,
            key=f"{method_key}_custom_description_{program_widget_key}",
        ).strip()

        custom_subtopic_text = st.text_area(
            "Enter 3–5 subtopics or concepts you understand well:",
            placeholder=(
                "Example: leadership styles, decision-making, "
                "organizational culture, communication, policy"
            ),
            height=80,
            key=f"{method_key}_custom_subtopics_{program_widget_key}",
        )

        custom_subtopics = parse_custom_items(custom_subtopic_text)

        return {
            "topic": custom_topic,
            "subtopics": custom_subtopics,
            "description": custom_description,
            "analogy": "",
            "source": "custom",
        }

    # --------------------------------------------------------
    # Predefined topic from the JSON file
    # --------------------------------------------------------

    topic_details = available_topics[selected_topic_option]

    predefined_subtopics = topic_details.get("subtopics", [])
    topic_description = topic_details.get("description", "")
    topic_analogy = topic_details.get("analogy", "")

    st.info(topic_description)

    # Automatically fill all five subtopics as comma-separated text.
    # The participant can add, remove, or modify the entries.
    default_subtopic_text = ", ".join(predefined_subtopics)

    edited_subtopic_text = st.text_area(
        "Review and edit the subtopics or concepts you understand well:",
        value=default_subtopic_text,
        help=(
            "The suggested subtopics are filled automatically. "
            "You may add, remove, or modify them. Keep 3–5 entries "
            "separated by commas."
        ),
        height=80,
        key=(
            f"{method_key}_editable_subtopics_"
            f"{program_widget_key}_{selected_topic_option}"
        ),
    )

    selected_subtopics = parse_custom_items(edited_subtopic_text)

    return {
        "topic": selected_topic_option,
        "subtopics": selected_subtopics,
        "description": topic_description,
        "analogy": topic_analogy,
        "source": "predefined",
    }

# ============================================================
# Home/onboarding page
# ============================================================

if st.session_state.page == "home":

    st.title("Grad Student Advisor Recommender System")

    if st.session_state.user_name == "":

        st.markdown(
            "### Please enter your information to get started:"
        )

        # ----------------------------------------------------
        # 1. Participant name
        # ----------------------------------------------------

        name_input = st.text_input(
            "Your name:",
            placeholder="Type your name here",
            key="user_name_input",
        )

        # ----------------------------------------------------
        # 2. Program selection
        # ----------------------------------------------------

        program_names = list(program_topic_data.keys())

        selected_program_option = st.selectbox(
            "What is your current or most recent program of study?",
            options=[""] + program_names + ["Other"],
            format_func=lambda value: (
                "Choose an option" if value == "" else value
            ),
            help=(
                "Select the closest program. Choose 'Other' if your "
                "program is not included."
            ),
            key="program_option",
        )

        if selected_program_option == "Other":

            selected_program = st.text_input(
                "Enter your program of study:",
                placeholder="Example: Sociology, Education, or Business",
                key="custom_program",
            ).strip()

            # There are no predefined topics for a custom program.
            selected_program_data = {}

        elif selected_program_option:

            selected_program = selected_program_option

            selected_program_data = program_topic_data[
                selected_program_option
            ]

        else:
            selected_program = ""
            selected_program_data = {}

        cosine_selection = empty_topic_selection()
        lda_selection = empty_topic_selection()

        # ----------------------------------------------------
        # 3. Show topics after a program has been selected
        # ----------------------------------------------------

        if selected_program_option:

           cosine_selection = render_topic_selection(
                    method_key="cosine",
                    topic_number=1,
                    program_data=selected_program_data,
                    program_widget_key=selected_program_option,
                )
                
           st.divider()
                
           lda_selection = render_topic_selection(
                    method_key="LDA",
                    topic_number=2,
                    program_data=selected_program_data,
                    program_widget_key=selected_program_option,
                )

        # ----------------------------------------------------
        # 4. Validate and store the information
        # ----------------------------------------------------

        if st.button("Start", type="primary"):

            validation_errors = []

            participant_name = name_input.strip()

            if not participant_name:
                validation_errors.append(
                    "Please enter your name."
                )

            if not selected_program:
                validation_errors.append(
                    "Please select or enter your program of study."
                )

            selections_to_validate = [
                ("Cosine Similarity", cosine_selection),
                ("LDA Topic Modeling", lda_selection),
            ]

            for method_label, selection in selections_to_validate:

                if not selection["topic"]:
                    validation_errors.append(
                        f"Please select or enter a topic for "
                        f"{method_label}."
                    )

                number_of_subtopics = len(selection["subtopics"])

                if number_of_subtopics < 3:
                    validation_errors.append(
                        f"Please select or enter at least three "
                        f"subtopics for {method_label}."
                    )

                elif number_of_subtopics > 5:
                    validation_errors.append(
                        f"Please use no more than five subtopics "
                        f"for {method_label}."
                    )

            # Display all validation problems.
            if validation_errors:

                for error_message in validation_errors:
                    st.warning(error_message)

            else:

                # --------------------------------------------
                # Main structured session-state object
                # --------------------------------------------

                background_profile = {
                    "program": selected_program,
                    "cosine": {
                        "topic": cosine_selection["topic"],
                        "subtopics": cosine_selection["subtopics"],
                        "description": cosine_selection["description"],
                        "analogy": cosine_selection["analogy"],
                        "source": cosine_selection["source"],
                    },
                    "LDA": {
                        "topic": lda_selection["topic"],
                        "subtopics": lda_selection["subtopics"],
                        "description": lda_selection["description"],
                        "analogy": lda_selection["analogy"],
                        "source": lda_selection["source"],
                    },
                }

                st.session_state.user_name = participant_name
                st.session_state.field_of_study = selected_program

                st.session_state.background_profile = (
                    background_profile
                )

                # --------------------------------------------
                # Individual Cosine session values
                # --------------------------------------------

                st.session_state.cosine_topic = (
                    cosine_selection["topic"]
                )

                st.session_state.cosine_subtopics = (
                    cosine_selection["subtopics"]
                )

                st.session_state.cosine_description = (
                    cosine_selection["description"]
                )

                st.session_state.cosine_analogy = (
                    cosine_selection["analogy"]
                )

                # --------------------------------------------
                # Individual LDA session values
                # --------------------------------------------

                st.session_state.lda_topic = (
                    lda_selection["topic"]
                )

                st.session_state.lda_subtopics = (
                    lda_selection["subtopics"]
                )

                st.session_state.lda_description = (
                    lda_selection["description"]
                )

                st.session_state.lda_analogy = (
                    lda_selection["analogy"]
                )

                # --------------------------------------------
                # Backward compatibility with old variables
                # --------------------------------------------

                st.session_state.specific_topics = (
                    f"Cosine: {cosine_selection['topic']}; "
                    f"LDA: {lda_selection['topic']}"
                )

                combined_subtopics = list(
                    dict.fromkeys(
                        cosine_selection["subtopics"]
                        + lda_selection["subtopics"]
                    )
                )

                st.session_state.background_keywords = ", ".join(
                    combined_subtopics
                )

                st.success(
                    f"Hello, {st.session_state.user_name}!"
                )

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
                                msg+='. Topic Similarity score: '+str(lda1['Score'][i])
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
        
            st.markdown(
                    """
                    <style>
                    .rec-table {
                        width: 100%;
                        table-layout: fixed;
                        border-collapse: collapse;
                    }
                
                    .rec-table th,
                    .rec-table td {
                        border: 1px solid #e6e6e6;
                        padding: 8px 10px;
                        vertical-align: top;
                        white-space: normal !important;
                        overflow-wrap: break-word;
                        word-wrap: break-word;
                        line-height: 1.55;
                    }
                
                    /* Center all column headings */
                    .rec-table th {
                        text-align: center;
                        font-weight: 600;
                        vertical-align: middle;
                    }
                
                    /* Justify table body text */
                    .rec-table td {
                        text-align: justify;
                        text-justify: inter-word;
                    }
                
                    /* Keep similarity score values centered */
                    .rec-table td:nth-child(3) {
                        text-align: center;
                    }
                
                    /* Column widths: Name | Keywords | Score | Publication | Affiliation */
                    .rec-table th:nth-child(1),
                    .rec-table td:nth-child(1) {
                        width: 9%;
                    }
                
                    .rec-table th:nth-child(2),
                    .rec-table td:nth-child(2) {
                        width: 26%;
                    }
                
                    .rec-table th:nth-child(3),
                    .rec-table td:nth-child(3) {
                        width: 7%;
                    }
                
                    .rec-table th:nth-child(4),
                    .rec-table td:nth-child(4) {
                        width: 34%;
                    }
                
                    .rec-table th:nth-child(5),
                    .rec-table td:nth-child(5) {
                        width: 24%;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
        
            # ---------------- Table 1: Text similarity ----------------
            st.write("Top 3 recommended advisors based on Text Similarity of keywords:")
        
            df1_new = df1[
                ["Name", "Keywords", "Similarity Score", "Publication", "Affiliation"]
            ].copy()
        
            df1_new["Similarity Score"] = df1_new["Similarity Score"].map(
                lambda x: f"{x:.4f}"
            )
        
            df1_new = df1_new.rename(
                columns={"Similarity Score": "Text Similarity"}
            )
        
            for col in ["Name", "Keywords", "Publication", "Affiliation"]:
                df1_new[col] = df1_new[col].map(clean_text)
        
            st.markdown(
                render_rec_table(df1_new),
                unsafe_allow_html=True
            )
        
            # ---------------- Table 2: LDA topic similarity ----------------
            topic_id = df3["Topic"].iloc[0]
            topic_words = df3["Words"].iloc[0]
        
            if isinstance(topic_words, list):
                topic_words = ", ".join(topic_words)
        
            st.write(
                    f"LDA generated 30 topics, and the most relevant topic was selected. "
                    f"Selected Topic: {topic_id}. "
                    f"Keywords for selected topic: {topic_words}"
                )
                
            
                
            st.caption(
                    "Note: Keywords are stemmed model terms. Some words may look shortened, such as "
                    "'softwar' for 'software' and 'qualiti' for 'quality'. They are not spelling errors."
                )
        
            st.write("Top 3 recommended advisors based on selected LDA Topic:")
        
            df2_new = df2[
                ["LDA_Name", "Keywords_LDA", "Score", "Publication", "Affiliation"]
            ].copy()
        
            df2_new = df2_new.rename(
                columns={
                    "LDA_Name": "Name",
                    "Keywords_LDA": "Keywords (LDA)",
                    "Score": "Topic Similarity"
                }
            )
        
            df2_new["Topic Similarity"] = df2_new["Topic Similarity"].map(
                lambda x: f"{x:.4f}"
            )
        
            for col in ["Name", "Keywords (LDA)", "Publication", "Affiliation"]:
                df2_new[col] = df2_new[col].map(clean_text)
        
            st.markdown(
                render_rec_table(df2_new),
                unsafe_allow_html=True
            )

            if st.session_state.page == "v5":
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
            if st.session_state.page == "v4":
                        render_v2(st.session_state.selected_scenarios[3])
                       
            if st.session_state.page == "v1":
                        render_v1(st.session_state.selected_scenarios[0])
            if st.session_state.page == "v3":
                        render_v1(st.session_state.selected_scenarios[2])
                        
                      
