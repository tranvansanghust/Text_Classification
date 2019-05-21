import numpy
import os
import logging
import utils
import pickle
import itertools
from operator import itemgetter

from gensim import corpora
from gensim import models
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from gensim.test.utils import datapath
from gensim.models import LdaModel

from preprocess import get_data, pre_process_doc

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
DICT_SIZE = 15000

class LDA():
    def __init__(self, model_file='./model/lda_model.model'):
        self.stopwords = utils.load_stopwords()
        self.num_topics = 50

        if os.path.isfile('./data/dictionary.pkl'):
            with open('./data/dictionary.pkl', 'rb') as f:
                self.dictionary = pickle.load(f)
            
            with open('./data/tfidf.pkl', 'rb') as f:
                self.tfidf = pickle.load(f)

        else:
            self.dictionary, self.tfidf = self.create_dictionary()

        if os.path.isfile(model_file):
            self.model = LdaModel.load(model_file)

        else:
            self.model = self.train(path_save=model_file)

    # Create dictionary
    def create_dictionary(self):
        print('Creating dictionary and tfidf ...')
        X_train, _ = get_data('./data/Train_Full', mode='from_file')
        texts = utils.remove_stopwords(X_train, self.stopwords)

        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.7, keep_n=DICT_SIZE)

        # create tfidf
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)

        # saving dictionary
        with open('./data/dictionary.pkl', 'wb') as f:
            pickle.dump(dictionary, f)
        
        with open('./data/tfidf.pkl', 'wb') as f:
            pickle.dump(tfidf, f)

        print('Created dictionary and tfidf.')

        return dictionary, tfidf

    # Create corpus
    def create_corpus(self, type_corpus='train', mode='create'):
        if mode == 'create':
            print('Creating ' + type_corpus + ' corpus ...')
            if type_corpus == 'train':
                X_train, y = get_data('./data/Train_Full', mode='from_file')
                texts = utils.remove_stopwords(X_train, self.stopwords)
                corpus = [self.dictionary.doc2bow(text) for text in texts]
            
            elif type_corpus == 'test':
                X_test, y = get_data('./data/Test_Full', mode='from_file')
                texts = utils.remove_stopwords(X_test, self.stopwords)
                corpus = [self.dictionary.doc2bow(text) for text in texts]
            
        elif mode == 'load':
            print('load corpus')
            with open('./data/' + type_corpus + '_corpus.pkl', 'rb') as f:
                corpus = pickle.load(f)

            with open('./data/y_' + type_corpus + '.pkl', 'rb') as f:
                y = pickle.load(f)

        return corpus, y
    
    def train(self, path_save='./model/lda_model.model', passes=5, chunksize=100, workers=3):
        print('Start training ...')
        # load corpus and filter dictionary
        train_corpus, _ = self.create_corpus(type_corpus='train', mode='load')
        corpus_tfidf = self.tfidf[train_corpus]

        lda_model = models.ldamulticore.LdaMulticore(corpus=corpus_tfidf, 
                                                    id2word=self.dictionary,
                                                    num_topics=self.num_topics, 
                                                    passes=passes, 
                                                    chunksize=chunksize,
                                                    per_word_topics=True, 
                                                    workers=workers,
                                                    eval_every=1)
        lda_model.save(path_save)

        return lda_model

    def get_feature_vec(self, type_data='train'):            
        corpus, text_labels = self.create_corpus(type_corpus=type_data, mode='load')

        vectors = []
        labels = []
        
        corpus_tfidf = self.tfidf[corpus]

        for i in range(len(corpus_tfidf)):
            top_topics = self.model.get_document_topics(corpus_tfidf[i], minimum_probability=0.0)
            topic_vec = [top_topics[j][1] for j in range(self.num_topics)]
            label = self.convert_to_onehot(text_labels[i])

            vectors.append(topic_vec)
            labels.append(label)

        return vectors, labels
    
    def cluster(self, doc_file):
        texts = pre_process_doc(doc_file)
        clean_text = utils.remove_stopwords(texts, self.stopwords)
        corpus = [self.dictionary.doc2bow(text) for text in clean_text]
        corpus_tfidf = self.tfidf[corpus]
        top_topics = self.model.get_document_topics(corpus_tfidf[0], minimum_probability=0.0)
        topic_vec = [top_topics[j][1] for j in range(self.num_topics)]

        return [topic_vec]
        
    def convert_to_onehot(self, label):
        if label == 'Chinh tri Xa hoi':
            return 0
        elif label == 'Doi song':
            return 1
        elif label == 'Khoa hoc':
            return 2
        elif label == 'Kinh doanh':
            return 3
        elif label == 'Phap luat':
            return 4
        elif label == 'Suc khoe':
            return 5
        elif label == 'The gioi':
            return 6
        elif label == 'The thao':
            return 7
        elif label == 'Van hoa':
            return 8
        elif label == 'Vi tinh':
            return 9

    def save_data(self):
        print('saving data.')
        train_corpus, train_id2word, y_train = self.load_data()
        with open('./data/Corpus_test.pkl', 'wb') as f:
            pickle.dump(train_corpus, f)
        
        with open('./data/Y_test.pkl', 'wb') as f:
            pickle.dump(y_train, f)

        with open('./data/test_id2word.pkl', 'wb') as f:
            pickle.dump(train_id2word, f)
        
        print('done save data')
    
    def load_pkl_data(self, type_data='train'):
        with open('./data/Corpus_' + type_data + '.pkl', 'rb') as f:
            train_corpus = pickle.load(f)
        
        with open('./data/Y_'+ type_data +'.pkl', 'rb') as f:
            y_train = pickle.load(f)

        with open('./data/'+ type_data +'_id2word.pkl', 'rb') as f:
            train_id2word = pickle.load(f)
        
        return train_corpus, train_id2word, y_train

    def predict(self):
        self.model.print_topics(15, num_words=20)

if __name__ == "__main__":
    lda = LDA()
