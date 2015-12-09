from __future__ import division
import cPickle
import flask
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn
from static.generate_features import *

#---------- MODEL IN MEMORY ----------------#

# load Poisson Regression Model for Share Count Predictions
with open('static/pois_regress.pkl', 'rb') as f:
    pois_reg = cPickle.load(f)

# load Random Forest Classification Model for Viarality Probability Predictions
with open('static/rf_class.pkl', 'rb') as f:
    RF_class = cPickle.load(f)

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    serve our visualization page, mashable_prediction.html
    """
    with open("mashable_prediction.html", 'r') as viz_file:
        return viz_file.read()

#recieve data from application, get results from analysis and send back to javascript
@app.route('/make_predict', methods=["POST"])
def make_predict():
    data = flask.request.json
    headline = data["headline"]
    content = data["content"].encode('utf8')
    tags = data["tags"]
    day_published = data["day_pub"]
    channel = data["channel"]
    num_imgs = int(data["num_imgs"])
    index = [0]
    columns = [u'LDA_0_prob', u'LDA_1_prob', u'LDA_2_prob', u'LDA_3_prob',
       u'LDA_4_prob', u'LDA_5_prob', u'LDA_6_prob', u'LDA_7_prob',
       u'LDA_8_prob', u'LDA_9_prob', u'average_token_length_content',
       u'average_token_length_title', u'avg_negative_polarity',
       u'avg_positive_polarity', u'data_channel_is_bus',
       u'data_channel_is_entertainment', u'data_channel_is_lifestyle',
       u'data_channel_is_socmed', u'data_channel_is_tech',
       u'data_channel_is_world', u'global_grade_level',
       u'global_rate_negative_words', u'global_rate_positive_words',
       u'global_reading_ease', u'global_sentiment_abs_polarity',
       u'global_sentiment_polarity', u'global_subjectivity', u'is_weekend',
       u'max_abs_polarity', u'max_negative_polarity', u'max_positive_polarity',
       u'min_negative_polarity', u'min_positive_polarity', u'n_tokens_content',
       u'n_tokens_title', u'num_imgs', u'num_tags', u'num_videos',
       u'r_non_stop_unique_tokens', u'r_non_stop_words', u'r_unique_tokens',
       u'rate_negative_words', u'rate_positive_words',
       u'title_sentiment_abs_polarity', u'title_sentiment_polarity',
       u'title_subjectivity', u'weekday_is_friday',
       u'weekday_is_monday', u'weekday_is_saturday', u'weekday_is_sunday',
       u'weekday_is_thursday', u'weekday_is_tuesday', u'weekday_is_wednesday']
    data_df = pd.DataFrame(index=index, columns=columns)
    create_metadata_fields(data_df, num_imgs, tags, day_published, channel)
    create_NLP_features(data_df, headline, content)
    create_lda_features(data_df, content)
    results = {}
    create_metadata_fields(results, num_imgs, tags, day_published, channel)
    create_NLP_features(results, headline, content)
    create_lda_features(results, content)
    results['est_shares'] = round(pois_reg.predict(sm.add_constant(data_df))[0],-2)
    results['est_prob'] = round(RF_class.predict_proba(sm.add_constant(data_df))[0][1],2)
    polarity_list = [(w.decode('utf-8'), TextBlob(w.decode('utf-8')).polarity) 
                      for w in "".join(c for c in content 
                                       if c not in string.punctuation).split()]
    polarity_data= [{'word': w, 'polarity': abs(p), 'color': '#1A79BB'} 
                     if p>0 
                     else {'word': w, 'polarity': abs(p), 'color': '#bb1a29'}
                     for (w,p) in polarity_list if p != 0]
    results['polarity_data'] = polarity_data              
    return flask.jsonify(results)

#recieve data from application, get results from recommendation and send back to javascript
@app.route('/get_recommend', methods=["POST"])
def get_recommend():
    data = flask.request.json
    headline = data["headline"]
    content = data["content"].encode('utf8')
    tags = data["tags"]
    day_published = data["day_pub"]
    channel = data["channel"]
    num_imgs = int(data["num_imgs"])
    index = [0]
    columns = [u'LDA_0_prob', u'LDA_1_prob', u'LDA_2_prob', u'LDA_3_prob',
       u'LDA_4_prob', u'LDA_5_prob', u'LDA_6_prob', u'LDA_7_prob',
       u'LDA_8_prob', u'LDA_9_prob', u'average_token_length_content',
       u'average_token_length_title', u'avg_negative_polarity',
       u'avg_positive_polarity', u'data_channel_is_bus',
       u'data_channel_is_entertainment', u'data_channel_is_lifestyle',
       u'data_channel_is_socmed', u'data_channel_is_tech',
       u'data_channel_is_world', u'global_grade_level',
       u'global_rate_negative_words', u'global_rate_positive_words',
       u'global_reading_ease', u'global_sentiment_abs_polarity',
       u'global_sentiment_polarity', u'global_subjectivity', u'is_weekend',
       u'max_abs_polarity', u'max_negative_polarity', u'max_positive_polarity',
       u'min_negative_polarity', u'min_positive_polarity', u'n_tokens_content',
       u'n_tokens_title', u'num_imgs', u'num_tags', u'num_videos',
       u'r_non_stop_unique_tokens', u'r_non_stop_words', u'r_unique_tokens',
       u'rate_negative_words', u'rate_positive_words',
       u'title_sentiment_abs_polarity', u'title_sentiment_polarity',
       u'title_subjectivity', u'weekday_is_friday',
       u'weekday_is_monday', u'weekday_is_saturday', u'weekday_is_sunday',
       u'weekday_is_thursday', u'weekday_is_tuesday', u'weekday_is_wednesday']
    data_df = pd.DataFrame(index=index, columns=columns)
    create_metadata_fields(data_df, num_imgs, tags, day_published, channel)
    create_NLP_features(data_df, headline, content)
    create_lda_features(data_df, content)
    results = {}
    create_metadata_fields(results, num_imgs, tags, day_published, channel)
    create_NLP_features(results, headline, content)
    create_lda_features(results, content)
    results['est_shares'] = round(pois_reg.predict(sm.add_constant(data_df))[0],-2)
    results['est_prob'] = round(RF_class.predict_proba(sm.add_constant(data_df))[0][1],2)
    data_df['weekday_is_monday'] = 0
    data_df['weekday_is_tuesday'] = 0
    data_df['weekday_is_wednesday'] = 0
    data_df['weekday_is_thursday'] = 0
    data_df['weekday_is_friday'] = 0
    data_df['weekday_is_saturday'] = 0
    data_df['weekday_is_sunday'] = 1
    data_df['is_weekend'] = 1
    results['est_shares_sun'] = round(pois_reg.predict(sm.add_constant(data_df))[0],-2)
    results['est_prob_sun'] = round(RF_class.predict_proba(sm.add_constant(data_df))[0][1],2)
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)

if __name__ == '__main__':
    app.run(debug=True,
        host = "0.0.0.0",
        port = 5000
    )