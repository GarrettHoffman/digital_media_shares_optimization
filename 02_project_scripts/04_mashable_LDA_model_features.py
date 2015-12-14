from gensim import corpora, models, similarities, matutils
import string
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pymongo

# connect to MongoDB collection
client = pymongo.MongoClient()
db = client.mashable
collection = client.mashable.articles

# pull article content from MongoDB
content = []
for doc in collection.find({}, {'_id': 0, 'content': 1}):
    content.append(doc['content'].encode('utf8'))

# check number of documents is equal to expected
len(content)

# define stop words to exclude from LDA topic modeling
stop = stopwords.words('english')

# remove punctuation
content_no_punc = ["".join(char for char in text
                           if char not in string.punctuation) 
                   for text in content]

# remove stopwords and tokenize
documents = [[word.decode('utf-8')
              for word in text.lower().split() 
              if word.decode('utf-8') not in stop] 
              for text in content_no_punc]

# define lemmatizer
lmtzr = WordNetLemmatizer()

# lemmatize vocabularly
documents = [[lmtzr.lemmatize(token) for token in doc]
              for doc in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for doc in documents:
     for token in doc:
            frequency[token] += 1

documents = [[token for token in doc if frequency[token] > 1]
              for doc in documents]

# create dictionary
dictionary = corpora.Dictionary(documents)
# store the dictionary, for future reference
dictionary.save('mashable_LDA_dictionary.dict')

# load dictionary
#dictionary = corpora.Dictionary.load('mashable_LDA_dictionary.dict')

# create corpus for model
corpus = [dictionary.doc2bow(doc) for doc in documents]

# store to disk, for later use
corpora.MmCorpus.serialize('mashable_LDA_corpara.mm', corpus) 

# load corpus
#corpus = corpora.MmCorpus('mashable_LDA_corpara.mm')

# train LDA model
# alpha and eta are hyperparameters that affect sparsity of the 
# document-topic (theta) and topic-word (lambda) distributions. 
# Both default to a symmetric 1.0/num_topics prior. Setting to 'auto'
# will learns an asymmetric prior directly from your data.

lda = models.LdaModel(corpus,
               id2word = dictionary,
               alpha = 'auto',
               eta = 'auto',
               num_topics=10)

# save model
lda.save('mashable.lda')

# load model
#lda = models.LdaModel.load('mashable.lda')

def get_lda_features(doc):
    
    """
    Pull document from MongoDB collection of Mashable Articles 
    and generate LDA topic probabilities.
    
    Arguments:
    Doc -- MongoDB Document containing Mashable content
    
    Output:
    Stores LDA topic probability results in Mongo DB for Document
    """
    
    # pull content from Mongo Doc
    content = doc['content'].encode('utf8')
    
    # remove punctuation
    content = "".join(char for char 
                      in content 
                      if char 
                      not in string.punctuation)
    
    # remove stopwords and tokenize
    content = [word.decode('utf-8')
               for word in content.lower().split() 
               if word.decode('utf-8') not in stop]
    
    # lemmatize vocabularly
    content = [lmtzr.lemmatize(token) 
               for token in content]
    
    # get LDA features for Model
    topic_probs = lda[dictionary.doc2bow(content)]
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
    
    collection.update_one({"_id": doc["_id"]}, 
                          {"$set": {"LDA_0_prob": LDA_topics[0], 
                                    "LDA_1_prob": LDA_topics[1],
                                    "LDA_2_prob": LDA_topics[2], 
                                    "LDA_3_prob": LDA_topics[3], 
                                    "LDA_4_prob": LDA_topics[4],
                                    "LDA_5_prob": LDA_topics[5], 
                                    "LDA_6_prob": LDA_topics[6],
                                    "LDA_7_prob": LDA_topics[7],
                                    "LDA_8_prob": LDA_topics[8],
                                    "LDA_9_prob": LDA_topics[9]}})

# get LDA features for all Mongo docs

progress_counter = 0 

for doc in collection.find({}, {"content": 1}):

    get_lda_features(doc)

    # show progress
    progress_counter += 1
    if progress_counter %100 == 0:
        print progress_counter