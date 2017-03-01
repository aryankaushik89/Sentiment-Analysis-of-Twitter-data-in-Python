import pandas as pd
import numpy as np

test_data_df = pd.read_csv('C:/apple_corpus/apple_test.csv', header=None, delimiter=",")
test_data_df.columns = ["Text"]    
train_data_df = pd.read_csv('C:/apple_corpus/apple_train.csv', header=None, delimiter=",")
train_data_df.columns = ["Sentiment","Text"]


import re, nltk
from sklearn.feature_extraction.text import CountVectorizer        
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems


vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 100
)

corpus_data_features = vectorizer.fit_transform(
train_data_df.Text.tolist() + test_data_df.Text.tolist())

corpus_data_features_nd = corpus_data_features.toarray()


vocab = vectorizer.get_feature_names()
#print (vocab)

# Sum up the counts of each vocabulary word
dist = np.sum(corpus_data_features_nd, axis=0)
    
# For each, print the vocabulary word and the number of times it 
# appears in the data set
#for tag, count in zip(vocab, dist):
#    print (count, tag)
from sklearn.cross_validation import train_test_split
    
# remember that corpus_data_features_nd contains all of our 
# original train and test data, so we need to exclude
# the unlabeled test entries

X_train, X_test, y_train, y_test  = train_test_split(
        corpus_data_features_nd[0:len(train_data_df)], 
        train_data_df.Sentiment,
        train_size=.80, 
        random_state=1234)

from sklearn.linear_model import LogisticRegression
    
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)


from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred))

# train classifier
log_model = LogisticRegression()
log_model = log_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)
    
# get predictions
test_pred = log_model.predict(corpus_data_features_nd[len(train_data_df):])
    
# sample some of them
import random
spl = random.sample(range(len(test_pred)), 100)
    
# print text and labels
for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    print (sentiment, text)



