from __future__ import division
import cPickle
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities, matutils
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
import string
from textstat.textstat import textstat
import statsmodels.api as sm
import sklearn

# load topic model processing tools and LDA model
stop = stopwords.words('english')
lmtzr = WordNetLemmatizer()
dictionary = corpora.Dictionary.load('static/mashable_LDA_dictionary.dict')
corpus = corpora.MmCorpus('static/mashable_LDA_corpara.mm')
lda = models.LdaModel.load('static/mashable.lda')

def create_metadata_fields(data, num_imgs, tags, day_published, channel):
    
    """
    Add metadata to DF or Dictionary of data to be input 
    to mashable models for prediction.

    Arguements:
    data: DateFrame or Dictionary to write data to
    num_imgs: integer representing number of images in content
    tags: string representing article tags
    day_published: string representing day of week content published on
    channel: string represeingtin mashable channel content is published on
    """
    
    # add values for content metadata
    data['num_imgs'] = num_imgs
    data['num_tags'] = len(tags.replace(' ','').split(","))
    data['num_videos'] = 0
    
    # add values for weekday published
    if day_published == 'Monday':
        data['weekday_is_monday'] = 1
    else:
        data['weekday_is_monday'] = 0
        
    if day_published == 'Tuesday':
        data['weekday_is_tuesday'] = 1
    else:
        data['weekday_is_tuesday'] = 0
        
    if day_published == 'Wednesday':
        data['weekday_is_wednesday'] = 1
    else:
        data['weekday_is_wednesday'] = 0
        
    if day_published == 'Thursday':
        data['weekday_is_thursday'] = 1
    else:
        data['weekday_is_thursday'] = 0
        
    if day_published == 'Friday':
        data['weekday_is_friday'] = 1
    else:
        data['weekday_is_friday'] = 0
        
    if day_published == 'Saturday':
        data['weekday_is_saturday'] = 1
    else:
        data['weekday_is_saturday'] = 0
        
    if day_published == 'Sunday':
        data['weekday_is_sunday'] = 1
    else:
        data['weekday_is_sunday'] = 0
        
    if day_published == 'Saturday' or day_published == 'Sunday':
        data['is_weekend'] = 1
    else:
        data['is_weekend'] = 0
    
    # add values for channel
    if channel == 'Business':
        data['data_channel_is_bus'] = 1
    else:
        data['data_channel_is_bus'] = 0
    
    if channel == 'Entertainment':
        data['data_channel_is_entertainment'] = 1
    else:
        data['data_channel_is_entertainment'] = 0
    
    if channel == 'Lifestyle':
        data['data_channel_is_lifestyle'] = 1
    else:
        data['data_channel_is_lifestyle'] = 0
        
    if channel == 'Social Media':
        data['data_channel_is_socmed'] = 1
    else:
        data['data_channel_is_socmed'] = 0
    
    if channel == 'Technology':
        data['data_channel_is_tech'] = 1
    else:
        data['data_channel_is_tech'] = 0
        
    if channel == 'World':
        data['data_channel_is_world'] = 1
    else:
        data['data_channel_is_world'] = 0


def create_NLP_features(data, headline, content):

    """
    Add NLP features to DF or Dictionary of data to be input 
    to mashable models for prediction.

    Arguements:
    data: DateFrame or Dictionary
    headline: string containing article headline
    content: string containing article content
    """

    # number of words in title
    data['n_tokens_title'] = len(headline.split())

    # subjectivity
    data['title_subjectivity'] = TextBlob(headline).subjectivity

    # polarity
    data['title_sentiment_polarity'] = round(TextBlob(headline).polarity,2)

    # absolute value polarirty
    data['title_sentiment_abs_polarity'] = abs(data['title_sentiment_polarity'])

    # average word length
    data['average_token_length_title'] = np.mean([len(w) for w 
                                          in "".join(c for c in headline 
                                                     if c not in string.punctuation).split()])

    #generate content features

    # number of words
    data['n_tokens_content'] = len([w for w in content.split()])

    # rate of unique words
    data['r_unique_tokens'] = round(len(set([w.lower().decode('utf-8')
                               for w 
                               in "".join(c for c in content 
                                          if c not in string.punctuation).split()]))/data['n_tokens_content'],2)

    # rate of non-stop word
    data['r_non_stop_words'] = len([w.lower().decode('utf-8') 
                            for w in "".join(c for c in content 
                                             if c not in string.punctuation).split() 
                            if w.decode('utf-8') 
                            not in stop])/data['n_tokens_content']

    # rate of unique non-stop word
    data['r_non_stop_unique_tokens'] = len(set([w.lower().decode('utf-8') 
                               for w in "".join(c for c in content 
                                                if c not in string.punctuation).split() 
                               if w.decode('utf-8')
                               not in stop]))/data['n_tokens_content']

    # average word length
    data['average_token_length_content'] = np.mean([len(w) for w 
                                            in "".join(c for c in content
                                                       if c not in string.punctuation).split()])

    # subjectivity
    data['global_subjectivity'] = TextBlob(content.decode('utf-8')).subjectivity

    # polarity
    data['global_sentiment_polarity'] = round(TextBlob(content.decode('utf-8')).polarity,2)

    # absolute polarity
    data['global_sentiment_abs_polarity'] = abs(data['global_sentiment_polarity'])

    # get polarity by word
    polarity_list = [(w.decode('utf-8'), TextBlob(w.decode('utf-8')).polarity) 
                             for w in "".join(c for c in content 
                                              if c not in string.punctuation).split()]

    # global positive word rate
    data['global_rate_positive_words'] = len([(w,p) 
                                      for (w,p) 
                                      in polarity_list 
                                      if p > 0])/len(polarity_list)

    # global negative word rate
    data['global_rate_negative_words'] = len([(w,p) 
                                      for (w,p) 
                                      in polarity_list 
                                      if p < 0])/len(polarity_list)

    # positive word rate (among non-nuetral words)
    if [(w,p) for (w,p) in polarity_list if p != 0]:
        data['rate_positive_words'] = len([(w,p) 
                                   for (w,p) 
                                   in polarity_list 
                                   if p > 0])/len([(w,p) 
                                                   for (w,p) 
                                                   in polarity_list 
                                                   if p != 0])
    else:
        data['rate_positive_words'] = 0

    # negative word rate (among non-nuetral words)
    if [(w,p) for (w,p) in polarity_list if p != 0]:
        data['rate_negative_words'] = len([(w,p) 
                                   for (w,p) 
                                   in polarity_list 
                                   if p < 0])/len([(w,p) 
                                                   for (w,p) 
                                                   in polarity_list 
                                                   if p != 0])

    else:
        data['rate_negative_words'] = 0 

    # average polarity of positive words
    if [p for (w,p) in polarity_list if p > 0]:
        data['avg_positive_polarity'] = np.mean([p for (w,p) 
                                         in polarity_list 
                                         if p > 0])
    else:
        data['avg_positive_polarity'] = 0

    # minimum polarity of positive words
    if [p for (w,p) in polarity_list if p > 0]:
        data['min_positive_polarity'] = min([p for (w,p) 
                                     in polarity_list 
                                     if p > 0])
    else:
        data['min_positive_polarity'] = 0

    # maximum polarity of positive words
    if [p for (w,p) in polarity_list if p > 0]:
        data['max_positive_polarity'] = max([p for (w,p) 
                                     in polarity_list 
                                     if p > 0])
    else: 
        data['max_positive_polarity'] = 0

    # average polarity of negative words
    if [p for (w,p) in polarity_list if p < 0]:
        data['avg_negative_polarity'] = np.mean([p for (w,p) 
                                         in polarity_list 
                                         if p < 0])
    else:
        data['avg_negative_polarity'] = 0

    # minimum polarity of negative words
    if [p for (w,p) in polarity_list if p < 0]:
        data['min_negative_polarity'] = min([p for (w,p) 
                                     in polarity_list 
                                     if p < 0])
    else:
        data['min_negative_polarity'] = 0

    # maximum polarity of negative words
    if [p for (w,p) in polarity_list if p < 0]:
        data['max_negative_polarity'] = max([p for (w,p) 
                                 in polarity_list 
                                 if p < 0])
    else:
        data['max_negative_polarity'] = 0

    # abs maximum polarity, sum of abs of max positive and abs of min negative polarity
    data['max_abs_polarity'] = data['max_positive_polarity'] + abs(data['min_negative_polarity'])

    # Flesch Reading Ease
    data['global_reading_ease'] = textstat.flesch_reading_ease(content.decode('utf-8'))

    # Flesch Kincaid Grade Level
    data['global_grade_level'] = textstat.flesch_kincaid_grade(content.decode('utf-8'))


def create_lda_features(data, content):

    """
    Add NLP features to DF or Dictionary of data to be input 
    to mashable models for prediction.

    Arguements:
    data: DateFrame or Dictionary
    content: string representing article content
    """

    # remove punctuation
    content_tmp = "".join(char for char 
                      in content 
                      if char 
                      not in string.punctuation)
    
    # remove stopwords and tokenize
    content_tmp = [word.decode('utf-8')
               for word in content_tmp.lower().split() 
               if word.decode('utf-8') not in stop]
    
    # lemmatize vocabularly
    content_tmp = [lmtzr.lemmatize(token) 
               for token in content_tmp]
    
    # get LDA features for Model
    topic_probs = lda[dictionary.doc2bow(content_tmp)]
    topics = {}
    topics = {topic for (topic,prob) in topic_probs}
    LDA_topics = dict()
    for i in range(10):
        if i in topics: 
            for (topic,prob) in topic_probs:
                if topic == i:
                    LDA_topics[i] = prob
        else:
            LDA_topics[i] = 0
    
    data['LDA_0_prob'] = LDA_topics[0] 
    data['LDA_1_prob'] = LDA_topics[1]
    data['LDA_2_prob'] = LDA_topics[2] 
    data['LDA_3_prob'] = LDA_topics[3] 
    data['LDA_4_prob'] = LDA_topics[4]
    data['LDA_5_prob'] = LDA_topics[5] 
    data['LDA_6_prob'] = LDA_topics[6]
    data['LDA_7_prob'] = LDA_topics[7]
    data['LDA_8_prob'] = LDA_topics[8]
    data['LDA_9_prob'] = LDA_topics[9]
