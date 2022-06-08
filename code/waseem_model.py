import json, os
# from preprocessor import tokenize
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from nltk.tokenize import TweetTokenizer


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


import string
table = str.maketrans('', '', string.punctuation)

def tokenize(tweet):
    stripped = [w.translate(table) for w in tweet.split()]
    tweet = " ".join(stripped).strip()
    #     print(tweet)
    return tweet.split()



label_map_original = {'none': 0, 'racism': 1, 'sexism': 2}
label_map_modified = {
    0: 0,  #NH
    1: 1,  #H
    2: 1  #H
}
hate_map = {0: "NON HATE", 1: "HATE"}

def load_data():
    texts = []
    labels = []
    file = os.path.join('data', 'waseem_data_norm.json')
    with open(file, 'r') as f:
        waseem_data = json.load(f)
    for each_tweet in waseem_data:
        tweet = tokenize(waseem_data[each_tweet]['text'].lower())
        texts.append(tweet)
        label_3 = int(waseem_data[each_tweet]['label_map'])
        hate_label = label_map_modified[label_3]
        labels.append(hate_label)
    return texts, labels

def train_model():
    texts, labels = load_data()
    X = vectorizer.fit_transform(texts)
    Y = labels
    X, Y = shuffle(X, Y, random_state=42)
    waseem_model.fit(X, Y)
    
def score_tweet(new_text='', print_=True): # for a single tweet
    if len(new_text) == 0:
        new_text = input()
    test_tweet = vectorizer.transform([tokenize(new_text.lower())])
    pred_class = hate_map[int(waseem_model.predict(test_tweet)[0])]
    if print_:
        print("Predicted class is: {}".format(pred_class))
    pred_probs = waseem_model.predict_proba(test_tweet)[0]
    if print_:
        print("Predicted Prob of HATE: {}".format(pred_probs[1]))
    return pred_class, pred_probs[1]
base_path='/opt/qmeng/DRAGNET_ICDM21/Dragnet/'
waseem_model = pickle.load(open(f'{base_path}waseem/waseem_model.pkl','rb'))
vectorizer = pickle.load(open(f'{base_path}waseem/vectorizer.pkl','rb'))
nltk_tweet_tokenizer = pickle.load(open(f'{base_path}waseem/nltk_tweet_tokenizer.pkl','rb'))

def score_set(test_set, only_hate=False):
    lower_set = []
    for tw in test_set:
        lower_set.append(tokenize(tw.lower()))
    X_test = vectorizer.transform(test_set)
    y_pred = waseem_model.predict_proba(X_test)
    if only_hate:
        return y_pred[:,1]
    else:
        return y_pred

if __name__ == '__main__':
    s = score_set(['Hi, good','fuck'], only_hate=True)
    print(s)
