import numpy
import os
import logging
import utils
from operator import itemgetter

from gensim import corpora
from gensim import models
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from gensim.test.utils import datapath
from gensim.models import LdaModel

from preprocess import get_data

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LDA():
    def __init__(self, model_file='./model/lda_model.model'):
        self.stopwords = utils.load_stopwords()
        self.train_corpus, self.train_id2word = self.load_data()

        if os.path.isfile(model_file):
            self.model = LdaModel.load(model_file)

        else:
            self.model = self.train(path_save=model_file)

    # Create dictionary
    def create_dictionary(self, data):
        return corpora.Dictionary(data)

    # Create corpus
    def create_corpus(self, id2word, data):
        return [id2word.doc2bow(text) for text in data]
    
    def load_data(self):
        # X_train, y_train = get_data(os.path.join(data_path, 'Train_Full'))
        X_train, y_train = get_data('./data/Fix_Bug_Train')
        # texts
        texts = utils.remove_stopwords(X_train, self.stopwords)

        # Get corpus
        train_id2word = self.create_dictionary(texts)
        train_corpus = self.create_corpus(train_id2word, texts)
        bigram_train = texts

        return train_corpus, train_id2word
    
    def train(self, path_save='./model/lda_model.model', num_topics=10, passes=5, chunksize=100, workers=3):
        lda_model = models.ldamulticore.LdaMulticore(corpus=self.train_corpus, 
                                                    id2word=self.train_id2word,
                                                    num_topics=num_topics, 
                                                    passes=passes, 
                                                    chunksize=chunksize,
                                                    per_word_topics=True, 
                                                    workers=workers,
                                                    eval_every=1)
        lda_model.save(path_save)

        return lda_model
    def get_feature_vec(self):
        train_vecs = []
        for i in range(len(self.train_corpus)):
            print(self.train_corpus[i])
            top_topics = self.model.get_document_topics(self.train_corpus[i], minimum_probability=0.0)
            print(top_topics)
            break
            train_vecs.append(top_topics)
        


    def predict(self):
        self.model.print_topics(15, num_words=20)

# load lda model
# fname = datapath("./model/lda_model.model")
lda = LDA()
lda.get_feature_vec()