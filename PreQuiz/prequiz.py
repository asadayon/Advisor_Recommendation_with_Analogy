def load_questions(who):
    if who=='Amina Rahman':
        return [
     {
      "question": "Why does the advisor recommendation system for Amina use both cosine similarity and LDA topic modeling?",
        "options": [            
            "To combine specific keyword matches with broader research theme alignment for robust recommendations",
            "To ensure all advisors have the same ranking in both models"
        ],
      "answer": "1"
    },
    {
      "question": "What does Core Cornelius’s cosine similarity score indicate for Amina’s research interests in mobile applications?",
        "options": [
            "No alignment with Amina’s research interests",
            "A strong alignment with Amina’s research interests"
        ],
      "answer": "2"
    },
    {
      "question": "If Amina adds 'wearable sensor' to her keywords, which advisor might rise to the top in cosine similarity?",
        "options": [
            "Oriana Riva",
            "Cory Cornelius"
        ],
      "answer": "2"
    }
]
    elif who=='Emily Zhang':
        return [
     {
       "question": "Why does the recommendation system use both cosine similarity and LDA topic modeling for Emily’s advisor recommendations?",
        "options": [   
            "Cosine similarity matches exact keywords, while LDA captures broader research themes.",
            "Both models produce identical results to confirm advisor rankings."
            
        ],
      "answer": "1"
    },
    {
      "question": "What does a cosine similarity score for Eleni Stroulia indicate about Emily’s research interests?",
        "options": [
            "There is no alignment between Emily’s keywords and Eleni Stroulia’s research profile.",
            "Emily’s research interests are highly aligned with Eleni Stroulia’s research profile."
           
        ],  
      "answer": "2"
    },
    
    {
      "question": "If Emily removes 'software quality' from her keywords how might this affect the cosine similarity ranking?",
        "options": [
            "Nico Zazworka would remain the top recommendation due to his focus on design debt.",
            "Shari Lawrence Pfleeger might drop in ranking, as 'software quality' is a key overlap with her profile."
        ],
      "answer": "2"
    }
]
    elif who=='David Chen':
        return [
    {
      "question": "Why does the system use both cosine similarity and LDA topic modeling for David Chen’s advisor recommendations?",
        "options": [
            "To increase computational complexity",
            "To provide a robust match by combining keyword and thematic alignment"
        ],
      "answer": "2"
    },
    {
      "question": "What does a cosine similarity score for Michita Imai indicate for David Chen?",
        "options": [
          "A strong alignment with David’s keyword vector",
          "No alignment with David’s research interests"
            
        ],,
      "answer": "1"
    },
    {
     "question": "If David Chen removes ‘rescue robot’ from his keywords, what is the likely impact on the cosine similarity rankings?",
        "options": [
            
            "Matsuno’s score will decrease, possibly dropping his rank",
            "Imai’s score will increase significantly"
         
        ],
      "answer": "1"
    }
]
    if who=='Sara Lee':
        return [
        
    {
      "question": "Why does the system combine cosine similarity and LDA topic modeling for Sara’s recommendations?",
        "options": [            
            "To capture both keyword overlap and thematic alignment for robust matches",
            "To reduce the number of advisors considered"
        ],
      "answer": "1"
    },
    {
      "question": "What does a cosine similarity score for J. A. Levin indicate for Sara’s interests?",
        "options": [
            "No alignment with Sara’s research interests",
            "Strong alignment due to closely matching keywords"
        ],
      "answer": "2"
    },
    {
      "question": "If Sara removes ‘web spam’ from her keywords, what might happen to Masashi Toyoda’s cosine similarity ranking?",
        "options": [
            "It would increase due to broader focus",
            "It would decrease due to loss of key overlap"
        ],
      "answer": "2"
    }
    ]
