# 02_project_scripts

This directory contains scripts and notebooks used for project implementation

**01_mashable_scrape.py**: scrape headline, content and # of tags from Mashable URLs and store in MongoDB database

**02_mashable_data_work.ipynb**: join scraped content data with a few fields ("meta-features") from original UCI dataset 

**03_mashable_NLP_feature_eng.py**: generate NLP features and store in MongoDB documents

**04_mashable_LDA_model_features.py**: generate LDA model and LDA probability distribution topics and store in MongoDB documents

**05_mashable_shares_modeling.ipynb**: create models for number of shares and probability of exceeding 90th percentile of shares