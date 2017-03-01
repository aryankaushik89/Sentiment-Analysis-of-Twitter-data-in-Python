from twython import Twython
import csv
import json
from nltk.twitter import Query, Streamer, Twitter, TweetViewer, TweetWriter, credsfromfile
from nltk.twitter.util import json2csv
from nltk.twitter.util import json2csv_entities
with open('C:/Users/Admin/data/mydir/aapl.json') as fp:
    json2csv_entities(fp, 'tweets.csv',
                        ['id', 'text'], 'place', ['name', 'country'])
