# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:48:54 2021

@author: steph
"""
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def clean_sentence(sentence, stopwords=False):
    """ Function to clean text, removing non alpha numeric characters and converting to lowercase.
        Optional argument to remove stopwords as well.
        
        args: sentence, stopwords (boolean indicator to indicate whether stopwords will be removed)
        returns: cleaned sentence """
        
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence) 
    
    if stopwords:
         sentence = remove_stopwords(sentence) 
    
    return sentence
                    
def get_cleaned_sentences(df, stopwords=False):    
    """ Function to clean sentences in a dataframe. Loops through each sentence and handles the cleaning 
        in the clean_sentence function.
        
        args: dataframe, stopwords (boolean indicator to indicate whether stopwords will be removed)
        returns: list of cleaned sentences """
    
    cleaned_sentences = []

    for index,row in df.iterrows():
        cleaned = clean_sentence(row["questions"], stopwords)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences

def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, FAQdf):
    """ Function to return the answer to the most relevant question to the user query that exists in the database.
        Impose a minimum of 0.5 for cosine similarity score in order to have a cutoff for relevance to assure that
        irrelevant answers aren't being given.'
        
        args: user query embedding, embedding of database of questions, dataframe of questions and answers
        returns: answer to user query. If cosine similarity score for most similar question to user query is less than 0.5
        it will return a message saying that there is no relevant answer in the database."""
        
    max_sim=-1
    index_sim=-1
    
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding.reshape(1, -1), question_embedding)[0][0]
        if sim > max_sim:
            max_sim = sim
            index_sim = index
    
    answer = FAQdf.loc[index_sim, "answers"]
    if max_sim > 0.5:
        return answer
    else:
        return "Sorry, we don't have anything relevant to your question."

#Load dataset and examine dataset, rename columns to questions and answers
df = pd.read_csv("qaSample.csv")

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
cleaned_sentences = get_cleaned_sentences(df,stopwords=False)
sent_bertphrase_embeddings = sbert_model.encode(cleaned_sentences)

online = True
while online:
    
    question_orig = input("Ask me a question: ")

    question = clean_sentence(question_orig, stopwords=False)       
    question_embedding = sbert_model.encode([question])[0].reshape(1, -1)
    
    print(retrieveAndPrintFAQAnswer(question_embedding, sent_bertphrase_embeddings, df))
    
    invalid = True
    
    while invalid:
        check = input("Did you have another question? Please type 'Yes' or 'No'.: ")
        
        if check.lower() == 'no':
            print("Thank you for your time. We hoped you got the answers you were looking for!")
            online, invalid = False, False
        elif check.lower() == 'yes':
            invalid = False
        else:
            print("Sorry, I didn't get that. Please type in a response as indicated.")
      
