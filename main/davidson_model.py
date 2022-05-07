import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
from nltk.stem.porter import *
import string
import re
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
# from preprocessor import tokenize as tt

base_path='/opt/qmeng/DRAGNET_ICDM21/Dragnet/'

stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
table = str.maketrans('', '', string.punctuation)
stemmer = PorterStemmer()

sentiment_analyzer = VS()

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, ' ', parsed_text)
    parsed_text = re.sub(mention_regex, ' ', parsed_text)
    return parsed_text

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'),
            parsed_text.count('HASHTAGHERE'))


def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  #Get text only

    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(
        float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59,
        1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(
        206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)),
        2)

    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [
        FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms,
        num_words, num_unique_terms, sentiment['neg'], sentiment['pos'],
        sentiment['neu'], sentiment['compound'], twitter_objs[2],
        twitter_objs[1], twitter_objs[0], retweet
    ]
    #features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

def basic_tokenize(tweet):
    stripped = [w.translate(table) for w in tweet.split()]
    tweet = " ".join(stripped).strip()
#     print(tweet)
    return tweet.split()

def load_data():
    texts = []
    labels = []
    file = os.path.join(base_path, 'data', 'davidson_data_norm.pkl')
    with open(file, "rb") as f:
        dv_data = pickle.load(f)
    for each_tweet in dv_data:
        tweet = dv_data[each_tweet]['text']
        texts.append(tweet)
        label_3 = int(dv_data[each_tweet]['label_map'])
        labels.append(label_3)
    return texts, labels


davidson_model = pickle.load(open(f'{base_path}davidson/davidson_model.pkl','rb'))
pos_vectorizer = pickle.load(open(f'{base_path}davidson/pos_vectorizer.pkl','rb'))

vectorizer = TfidfVectorizer(tokenizer=basic_tokenize,
                             preprocessor=preprocess,
                             ngram_range=(1, 3),
                             stop_words=stopwords,
                             use_idf=True,
                             smooth_idf=False,
                             norm=None,
                             decode_error='replace',
                             max_features=10000,
                             min_df=5,
                             max_df=0.75)

vectorizer.fit(load_data()[0])

def score_set(test_set, only_hate=False):
    tweets_tags = []
    for tweet in test_set:
        tokens = basic_tokenize(preprocess(tweet))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweets_tags.append(tag_str)
    pos_features = pos_vectorizer.transform(pd.Series(tweets_tags)).toarray()
    tfidf_features = vectorizer.transform(test_set).toarray()
    other_features_ = get_feature_array(test_set)
    M_test = np.concatenate([tfidf_features,pos_features,other_features_],axis=1)
    X_test = pd.DataFrame(M_test)
    y_pred=davidson_model.predict_proba(X_test)
    if only_hate:
        return y_pred[:,1]
    else:
        return y_pred

if __name__ == '__main__':
    s = score_set(['Hi, good','fuck'], only_hate=True)
    print(s)