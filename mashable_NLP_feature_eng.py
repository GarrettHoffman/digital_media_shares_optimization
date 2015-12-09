from __future__ import division
import numpy as np
import pymongo
import nltk
from textblob import TextBlob
import string
from nltk.corpus import stopwords
from textstat.textstat import textstat

def engineer_NLP_features(doc):

    # get article headline and article content from Mongo DB document

    headline = doc['title']
    content = doc['content'].encode('utf-8')

    # generate headline features

    # number of words in title
    n_tokens_title = len(headline.split())

    # subjectivity
    title_subjectivity = TextBlob(headline).subjectivity

    # polarity
    title_sentiment_polarity = TextBlob(headline).polarity

    # absolute value polarirty
    title_sentiment_abs_polarity = abs(title_sentiment_polarity)

    # average word length
    average_token_length_title = np.mean([len(w) for w 
                                          in "".join(c for c in headline 
                                                     if c not in string.punctuation).split()])

    #generate content features

    # number of words
    n_tokens_content = len([w for w in content.split()])

    # rate of unique words
    r_unique_tokens = len(set([w.lower().decode('utf-8')
                               for w 
                               in "".join(c for c in content 
                                          if c not in string.punctuation).split()]))/n_tokens_content

    # rate of non-stop word
    r_non_stop_words = len([w.lower().decode('utf-8') 
                            for w in "".join(c for c in content 
                                             if c not in string.punctuation).split() 
                            if w.decode('utf-8') 
                            not in stop])/n_tokens_content

    # rate of unique non-stop word
    r_non_stop_unique_tokens = len(set([w.lower().decode('utf-8') 
                               for w in "".join(c for c in content 
                                                if c not in string.punctuation).split() 
                               if w.decode('utf-8')
                               not in stop]))/n_tokens_content

    # average word length
    average_token_length_content = np.mean([len(w) for w 
                                            in "".join(c for c in content
                                                       if c not in string.punctuation).split()])

    # subjectivity
    global_subjectivity = TextBlob(content.decode('utf-8')).subjectivity

    # polarity
    global_sentiment_polarity = TextBlob(content.decode('utf-8')).polarity

    # absolute polarity
    global_sentiment_abs_polarity = abs(global_sentiment_polarity)

    # get polarity by word
    polarity_list = [(w.decode('utf-8'), TextBlob(w.decode('utf-8')).polarity) 
                     for w in "".join(c for c in content 
                                      if c not in string.punctuation).split()]

    # global positive word rate
    global_rate_positive_words = len([(w,p) 
                                      for (w,p) 
                                      in polarity_list 
                                      if p > 0])/len(polarity_list)

    # global negative word rate
    global_rate_negative_words = len([(w,p) 
                                      for (w,p) 
                                      in polarity_list 
                                      if p < 0])/len(polarity_list)

    # positive word rate (among non-nuetral words)
    if [(w,p) for (w,p) in polarity_list if p != 0]:
        rate_positive_words = len([(w,p) 
                                   for (w,p) 
                                   in polarity_list 
                                   if p > 0])/len([(w,p) 
                                                   for (w,p) 
                                                   in polarity_list 
                                                   if p != 0])
    else:
        rate_positive_words = 0

    # negative word rate (among non-nuetral words)
    if [(w,p) for (w,p) in polarity_list if p != 0]:
        rate_negative_words = len([(w,p) 
                                   for (w,p) 
                                   in polarity_list 
                                   if p < 0])/len([(w,p) 
                                                   for (w,p) 
                                                   in polarity_list 
                                                   if p != 0])

    else:
       rate_negative_words = 0 

    # average polarity of positive words
    if [p for (w,p) in polarity_list if p > 0]:
        avg_positive_polarity = np.mean([p for (w,p) 
                                         in polarity_list 
                                         if p > 0])
    else:
        avg_positive_polarity = 0

    # minimum polarity of positive words
    if [p for (w,p) in polarity_list if p > 0]:
        min_positive_polarity = min([p for (w,p) 
                                     in polarity_list 
                                     if p > 0])
    else:
        min_positive_polarity = 0

    # maximum polarity of positive words
    if [p for (w,p) in polarity_list if p > 0]:
        max_positive_polarity = max([p for (w,p) 
                                     in polarity_list 
                                     if p > 0])
    else: 
        max_positive_polarity = 0

    # average polarity of negative words
    if [p for (w,p) in polarity_list if p < 0]:
        avg_negative_polarity = np.mean([p for (w,p) 
                                         in polarity_list 
                                         if p < 0])
    else:
        avg_negative_polarity = 0

    # minimum polarity of negative words
    if [p for (w,p) in polarity_list if p < 0]:
        min_negative_polarity = min([p for (w,p) 
                                     in polarity_list 
                                     if p < 0])
    else:
        min_negative_polarity = 0

    # maximum polarity of negative words
    if [p for (w,p) in polarity_list if p < 0]:
        max_negative_polarity = max([p for (w,p) 
                                 in polarity_list 
                                 if p < 0])
    else:
        max_negative_polarity = 0

    # abs maximum polarity, sum of abs of max positive and abs of min negative polarity
    max_abs_polarity = max_positive_polarity + abs(min_negative_polarity)

    # Flesch Reading Ease
    global_reading_ease = textstat.flesch_reading_ease(content.decode('utf-8'))

    # Flesch Kincaid Grade Level
    global_grade_level = textstat.flesch_kincaid_grade(content.decode('utf-8'))

    collection.update_one({"_id": doc["_id"]}, 
                          {"$set": {"n_tokens_title": n_tokens_title, 
                                    "title_subjectivity": title_subjectivity,
                                    "title_sentiment_polarity": title_sentiment_polarity,
                                    "title_sentiment_abs_polarity": title_sentiment_abs_polarity,
                                    "average_token_length_title": average_token_length_title,
                                    "n_tokens_content": n_tokens_content,
                                    "r_unique_tokens": r_unique_tokens,
                                    "r_non_stop_words": r_non_stop_words,
                                    "r_non_stop_unique_tokens": r_non_stop_unique_tokens,
                                    "average_token_length_content": average_token_length_content,
                                    "global_subjectivity": global_subjectivity,
                                    "global_sentiment_polarity": global_sentiment_polarity,
                                    "global_sentiment_abs_polarity": global_sentiment_abs_polarity,
                                    "global_rate_positive_words": global_rate_positive_words,
                                    "global_rate_negative_words": global_rate_negative_words,
                                    "rate_positive_words": rate_positive_words,
                                    "rate_negative_words": rate_negative_words,
                                    "avg_positive_polarity": avg_positive_polarity,
                                    "min_positive_polarity": min_positive_polarity,
                                    "max_positive_polarity": max_positive_polarity,
                                    "avg_negative_polarity": avg_negative_polarity,
                                    "min_negative_polarity": min_negative_polarity,
                                    "max_negative_polarity": max_negative_polarity,
                                    "max_abs_polarity": max_abs_polarity,
                                    "global_reading_ease": global_reading_ease,
                                    "global_grade_level": global_grade_level}})

stop = stopwords.words('english')

# connect to mongo db collection
client = pymongo.MongoClient()
db = client.mashable
collection = client.mashable.articles

progress_counter = 0 

for doc in collection.find({}, {"title": 1, "content": 1}):

    engineer_NLP_features(doc)

    # show progress
    progress_counter += 1
    print progress_counter