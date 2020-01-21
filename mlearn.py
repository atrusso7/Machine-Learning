import requests
import logging
import base64
import time
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import metrics

# Reading csv file of binary samples with their corresponding answers. Then assigning columns to their appropriate variables.
df = pd.read_csv('binary_data_train.csv')
hex_train = df['Binary']
target_train = df['Arch']

# I utilized the tutorial to assist with tokenizing the data
vec_opts = {
    "ngram_range": (1, 4),  # allow n-grams of 1-4 words in length (32-bits)
    "analyzer": "word",     # analyze hex words
    "token_pattern": "..",  # treat two characters as a word (e.g. 4b)
    "min_df": 3,          # for demo purposes, be very selective about features
}
v = CountVectorizer(**vec_opts)
X = v.fit_transform(hex_train, target_train)

#perform tfidf transform
idf_opts = {"use_idf": True}
idf = TfidfTransformer(**idf_opts)

# Adjusted the hyper-parameters of the classifier but didn't improve much from defaults
clf_opts = {"kernel": "linear"} 
clf = svm.SVC(**clf_opts)

# Perform the idf transform
X = idf.fit_transform(X)

# Created pipeline for ease of testing
pipeline = Pipeline([
    ('vec',  CountVectorizer(**vec_opts)),
    ('idf',  TfidfTransformer(**idf_opts)),
    ('clf',  svm.SVC(**clf_opts))
])

# Training my algorithm
X = pipeline.fit(hex_train, target_train)

# Mostly pulled off Praetorian Github
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Used to document session ID
cookie = 'a'

class Server(object):
    url = 'https://mlb.praetorian.com'
    log = logging.getLogger(__name__)

    def __init__(self):
        self.session = requests.session()
        self.binary  = None
        self.hash    = None
        self.wins    = 0
        self.targets = []

    def _request(self, route, method='get', data=None):
        global cookie
        while True:
            try:
                if method == 'get':
                    r = self.session.get(self.url + route)
                    # Only used to document session ID
                    cookie = r.cookies.get_dict()
                else:
                    r = self.session.post(self.url + route, data=data)
                if r.status_code == 429:
                    raise Exception('Rate Limit Exception')
                if r.status_code == 500:
                    raise Exception('Unknown Server Exception')

                return r.json()
            except Exception as e:
                self.log.error(e)
                self.log.info('Waiting 60 seconds before next request')
                time.sleep(60)

    def get(self):
        r = self._request("/challenge")
        self.targets = r.get('target', [])
        self.binary  = base64.b64decode(r.get('binary', ''))
        return r

    def post(self, target):
        r = self._request("/solve", method="post", data={"target": target})
        self.wins = r.get('correct', 0)
        self.hash = r.get('hash', self.hash)
        self.ans  = r.get('target', 'unknown')
        return r

if __name__ == "__main__":

    # create the server object
    s = Server()

    for _ in range(10000000):
        # query the /challenge endpoint
        # assigned variable for ease of sample gathering
        s.get()
        binary_blob = s.get()['binary']

        # Predict target and /solve 
        predict = X.predict([str(binary_blob)])
        target = predict[0]
        s.post(target)
        answer = s.ans

        # Return the binary with the correct answer
        s.log.info("Guess:[{: >9}]   Answer:[{: >9}]   Wins:[{: >3}]".format(target, s.ans, s.wins))
        
        # Used to build archive of samples
        row = [(str(binary_blob), str(answer))]
        with open('binary_data_learning.csv', 'a+') as csvFile:
             writer = csv.writer(csvFile)
             writer.writerows(row)
        csvFile.close()
        
        # Tried to incorporate a way to capture the hash but didn't work
        if s.hash:
            s.log.info("You win! {}".format(s.hash))
            the_hash = [str(s.hash)]
            with open('binary_data_hash.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(the_hash)
            csvFile.close()
            break