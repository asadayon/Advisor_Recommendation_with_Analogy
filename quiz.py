import streamlit as st
import random
import matplotlib.pyplot as plt
import mysql.connector
from mysql.connector import Error

def load_questions():
    return [
        {
            "question": "How can text similarity-based models be used in an advisor recommender system?",
            "options": [
                "By analyzing the personality traits of students and advisors to find the best match",
                "By comparing the textual descriptions of research interests from students and advisors to find the best alignment",
                "By ranking advisors based on their publication count and matching them with students",
                "By considering the geographical location of students and advisors for recommendations"
            ],
            "answer": "By comparing the textual descriptions of research interests from students and advisors to find the best alignment",
            "question_number":1
        },
        {
            "question": "What is the primary goal of using cosine similarity in advisor recommendation systems?",
            "options": [
                "To find the shortest path between two research interests",
                "To measure the similarity between the research interest keywords of students and advisors",
                "To classify research interests into predefined categories",
                "To generate random advisor-student pairings"
            ],
            "answer": "To measure the similarity between the research interest keywords of students and advisors",
            "question_number":2
        },
        {
            "question": "How is cosine similarity calculated between two vectors representing research interests?",
            "options": [
                "By adding the magnitudes of the two vectors",
                "By finding the difference between the two vectors",
                "By dividing the dot product of the vectors by the product of their magnitudes",
                "By calculating the Euclidean distance between the vectors"
            ],
            "answer": "By dividing the dot product of the vectors by the product of their magnitudes",
            "question_number":3
        },
        {
            "question": "Which of the following is true about cosine similarity when used in text similarity search for advisor recommendation?",
            "options": [
                "It only works with binary data",
                "It ranges from -1 to 1, where 1 means completely dissimilar",
                "It is not affected by the length of the vectors",
                "It is computationally intensive and not suitable for large datasets"
            ],
            "answer": "It is not affected by the length of the vectors",
            "question_number":4
        },
        {
            "question": "In the context of advisor recommendation, why is it important to preprocess the research interest keywords before applying cosine similarity?",
            "options": [
                "To increase the complexity of the algorithm",
                "To ensure that all keywords are in a uniform format for accurate comparison",
                "To reduce the dimensionality of the vectors",
                "To encrypt the data for security"
            ],
            "answer": "To ensure that all keywords are in a uniform format for accurate comparison",
            "question_number":5
        },
        {
            "question": "Which preprocessing step is commonly applied to research interest keywords before computing cosine similarity?",
            "options": [
                "Tokenization",
                "Vector normalization",
                "Stop word removal",
                "All of the above"
            ],
            "answer": "All of the above",
            "question_number":6
        },
        {
            "question": "What does a cosine similarity score of 1 indicate in the context of advisor recommendation?",
            "options": [
                "The research interests of the student and advisor are identical",
                "The research interests of the student and advisor are completely different",
                "The research interests of the student and advisor are orthogonal",
                "The cosine similarity score is invalid for this context"
            ],
            "answer": "The research interests of the student and advisor are identical",
            "question_number":7
        },
        {
            "question": "Why might cosine similarity be preferred over other similarity measures like Euclidean distance for text similarity in advisor recommendation systems?",
            "options": [
                "Because it is less affected by the size of the vectors",
                "Because it considers the absolute positions of words",
                "Because it only works with large datasets",
                "Because it is easier to interpret than other measures"
            ],
            "answer": "Because it is less affected by the size of the vectors",
            "question_number":8
        },
        {
            "question": "Consider the following list of all research interests: [Artificial Intelligence, Machine Learning, Data Mining, Natural Language Processing, Computer Vision, Robotics]. A student has the following research interests: [Artificial Intelligence, Machine Learning, Natural Language Processing, Computer Vision]. What is the vector representation for the student's research interests?",
            "options": [
                "[1,1,1,0,1,0]",
                "[1,1,0,1,1,0]",
                "[0,1,0,1,1,1]",
                "[1,0,1,1,0,1]"
            ],
            "answer": "[1,1,0,1,1,0]",
            "question_number":9
        },
        {
            "question": "Suppose student's research interest vector is: [1,0,1,1,0,0]. Given the following research interest vectors for four advisors: Advisor A: [1,1,1,0,1,0], Advisor B: [1,1,0,1,1,0], Advisor C: [0,1,0,1,1,1], Advisor D: [1,0,1,1,0,1]. Which advisor is most similar to the student based on cosine similarity?",
            "options": [
                "Advisor A",
                "Advisor B",
                "Advisor C",
                "Advisor D"
            ],
            "answer": "Advisor D",
            "question_number":10
        },
        {
            "question": "Suppose Student A has research interest keywords represented by the vector [1,0,1,1] and Advisor X has research interest keywords represented by the vector [0,1,1,1]. What is the cosine similarity between Student A and Advisor X?",
            "options": [
                "0.2",
                "0",
                "0.67",
                "1.0"
            ],
            "answer": "0.67",
            "question_number":11
        },
        {
            "question": "In the context of topic similarity search, what is a 'topic'?",
            "options": [
                "A specific research paper.",
                "A set of keywords representing a particular research area.",
                "The title of a student's thesis.",
                "A list of publications by an advisor."
            ],
            "answer": "A set of keywords representing a particular research area.",
            "question_number":12
        },
        {
            "question": "How does Latent Dirichlet Allocation (LDA) help in topic similarity search?",
            "options": [
                "By classifying documents into predefined categories.",
                "By calculating the sentiment of documents.",
                "By predicting the next word in a sentence.",
                "By identifying underlying topics in a collection of documents."
            ],
            "answer": "By identifying underlying topics in a collection of documents.",
            "question_number":13
        },
        {
            "question": "What type of input data is required for topic similarity search using LDA in advisor recommendation?",
            "options": [
                "Numerical data representing student grades.",
                "Keywords representing research interests of students and advisors.",
                "Images of research facilities.",
                "Audio recordings of lectures."
            ],
            "answer": "Keywords representing research interests of students and advisors.",
            "question_number":14
        },
        {
            "question": "Why is it important to preprocess the research interest keywords before applying LDA?",
            "options": [
                "To reduce noise and irrelevant information.",
                "To correct grammatical errors.",
                "To increase the size of the dataset.",
                "To ensure all keywords are in uppercase."
            ],
            "answer": "To reduce noise and irrelevant information.",
            "question_number":15
        },
        {
            "question": "What is the output of the LDA model in the context of advisor recommendation?",
            "options": [
                "A single topic label for each keyword.",
                "A ranked list of advisors for each student.",
                "A probability distribution of topics for each document.",
                "A list of recommended research papers."
            ],
            "answer": "A probability distribution of topics for each document.",
            "question_number":16
        },
        {
            "question": "What is a potential challenge when using LDA for advisor recommendation?",
            "options": [
                "LDA requires labeled data for training.",
                "Determining the optimal number of topics.",
                "LDA is only effective with small datasets.",
                "Ensuring all keywords are unique."
            ],
            "answer": "Determining the optimal number of topics.",
            "question_number":17
        },
        {
            "question": "In the context of LDA, why is it important to choose an appropriate number of topics?",
            "options": [
                "To ensure the model runs faster.",
                "To minimize the number of documents needed.",
                "To balance between topic specificity and generality.",
                "To increase the number of keywords in each document."
            ],
            "answer": "To balance between topic specificity and generality.",
            "question_number":18
        },
        {
            "question": "How does LDA represent each document in the corpus?",
            "options": [
                "As a single topic.",
                "As a mixture of topics with different proportions.",
                "As a random collection of words.",
                "As a sequence of characters."
            ],
            "answer": "As a mixture of topics with different proportions.",
            "question_number":19
        },
        {
            "question": "If LDA identifies that a student's research document has a topic proportion of [0.6,0.2,0.2], what can be inferred?",
            "options": [
                "The student is mostly interested in Topic 1.",
                "The student has equal interest in all topics.",
                "The student is least interested in Topic 1.",
                "The student’s interests are not related to any identified topics."
            ],
            "answer": "The student is mostly interested in Topic 1.",
            "question_number":20
        },
        {
            "question": "A student's topic distribution is [0.25,0.25,0.5]. Which advisor's topic distribution would most likely be a good match?",
            "options": [
                "[0.2,0.8,0.5]",
                "[0.4,0.4,0.2]",
                "[0.1,0.5,0.4]",
                "[0.3,0.2,0.5]"
            ],
            "answer": "[0.3,0.2,0.5]",
            "question_number":21
        },
        {
            "question": "If a word has the following distribution across topics: [0.3,0.4,0.3] and it appears 10 times in a document, how many times is it expected to belong to Topic 2?",
            "options": [
                "3",
                "4",
                "5",
                "6"
            ],
            "answer": "4",
            "question_number":22
        }
    ]
# Shuffle questions and options once before loading
if "shuffled_questions" not in st.session_state:
    questions = load_questions()
    random.shuffle(questions)
    for question in questions:
        random.shuffle(question["options"])
    st.session_state.shuffled_questions = questions

def connect_to_db():
    return mysql.connector.connect(
        host=st.secrets["HOST"],
        port=st.secrets["PORT"],
        user=st.secrets["USER"],
        password=st.secrets["PASSWORD"],
        database=st.secrets["DATABASE"]  # Replace with your actual database name
    )        
      
    



def run_quiz():
    questions = st.session_state.shuffled_questions

    st.title("Multiple Choice Quiz")

    user_answers = []
    correct_count = 0

    for i, question in enumerate(questions):
        st.write(f"Q{i+1}: {question['question']}")

        user_answer = st.radio(
            "Choose your answer:",
            question["options"],
            index=None,  # No option selected initially
            key=f"question_{i}"
        )
        user_answers.append(user_answer)
        
    name=st.text_input('Name: ')
    attempt_num=st.text_input('Attempt Number:')
    connection = connect_to_db()
    cursor = connection.cursor()
    if st.button("Submit"):
        for i, question in enumerate(questions):
            if user_answers[i] == question['answer']:
                correct_count += 1
                st.write(f"Q{i+1}: {question['question']}")
                st.success(f"Your answer: {user_answers[i]}", icon="✔")
            else:
                st.write(f"Q{i+1}: {question['question']}")
                st.error(f"Your answer: {user_answers[i]}", icon="❌")

        total_questions = len(questions)
        score_percentage = (correct_count / total_questions) * 100
        cursor.execute("INSERT INTO QuizAttempts (user_id, attempt_number, final_score) VALUES (%s, %s, %s)", (name, attempt_num, correct_count))
        attempt_id=cursor.lastrowid
        for i, question in enumerate(questions):
            cursor.execute("INSERT INTO AttemptDetails (attempt_id, question_number, chosen_option, is_correct) VALUES (%s, %s, %s, %s)",
                   (attempt_id, question['question_number'], user_answers[i], user_answers[i] == question['answer']))
        
        
        # Display score percentage bar
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.barh(0, score_percentage, color='green', height=0.5)
        ax.barh(0, 100 - score_percentage, left=score_percentage, color='red', height=0.5)
        ax.set_xlim(0, 100)
        ax.axis('off')

        # Annotate the percentage under the bar
        plt.text(50, -0.3, f"Score: {score_percentage:.2f}%", ha='center', va='center', fontsize=12)

        st.pyplot(fig)

        st.write(f"Your raw score is: {correct_count}/{total_questions}")
        st.write(f"Your percentile score is: {score_percentage:.2f}%")
        connection.commit()
        cursor.close()
        connection.close()

# Run the quiz
run_quiz()
