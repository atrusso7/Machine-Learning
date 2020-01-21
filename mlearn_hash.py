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

# Make numpy values easier to read.
# np.set_printoptions(precision=3, suppress=True)

# df = pd.read_csv('binary_data_train.csv')
# hex_train = df['Binary']
# target_train = df['Arch']

# df2 = pd.read_csv('binary_data_test.csv')
# hex_test2 = df2['Binary']
# target_test2 = df2['Arch']

# vec_opts = {
#     "ngram_range": (1, 4),  # allow n-grams of 1-4 words in length (32-bits)
#     "analyzer": "word",     # analyze hex words
#     "token_pattern": "..",  # treat two characters as a word (e.g. 4b)
#     "min_df": 3,          # for demo purposes, be very selective about features
# }
# v = CountVectorizer(**vec_opts)
# X = v.fit_transform(hex_train, target_train)

# idf_opts = {"use_idf": True}
# idf = TfidfTransformer(**idf_opts)

# clf_opts = {"kernel": "linear"} 
# clf = svm.SVC(**clf_opts)

# # perform the idf transform
# X = idf.fit_transform(X)

# pipeline = Pipeline([
#     ('vec',  CountVectorizer(**vec_opts)),
#     ('idf',  TfidfTransformer(**idf_opts)),
#     ('clf',  svm.SVC(**clf_opts))
# ])
# single = ['AAAIOUAAAC+KAABAvf/wOSkAAUgAAAg5IAAAL4kAAEC9/+Q5CAABL4gAAkCd/+w5QAARPSAAADkpAACRSQAMSA==']
# X = pipeline.fit(hex_train, target_train)

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Server(object):
    url = 'https://mlb.praetorian.com/hash/'
    log = logging.getLogger(__name__)

    def __init__(self):
        self.session = requests.session()
        self.binary  = None
        self.hash    = None
        self.wins    = 0
        self.targets = []

    def _request(self, route, method='get', data={'email':'atrusso7@gmail.com'}):
        while True:
            try:
                if method == 'get':
                    r = self.session.get(self.url + route)
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
    email = 'atrusso7@gmail.com'
    s = Server()
    print(s.hash)
    if s.hash:
            s.log.info("You win! {}".format(s.hash))