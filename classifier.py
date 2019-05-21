import numpy as np
import pickle
import os

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, accuracy_score
from LDA import LDA
from loadmodel import LoadModel

class Classifier():    
    def __init__(self, type_model='SVM'):
        self.type_model = type_model
        self.model_name = './model/classifier_' + self.type_model + '.sav'

        if os.path.isfile(self.model_name):
            with open(self.model_name, 'rb') as f:
                self.model = pickle.load(f)

        else:
            self.model = self.buil_model()
        

    def buil_model(self):
        if self.type_model == 'SVM':
            model = SGDClassifier(max_iter=1000,
                                tol=0.0001,
                                alpha=0.0001,
                                loss='modified_huber',
                                class_weight=None,
                                learning_rate='adaptive',
                                eta0=0.2)
        
        elif self.type_model == 'LogisticRegression':
            model = LogisticRegression(class_weight= None,
                                        solver='newton-cg',
                                        fit_intercept=True)
        
        elif self.type_model == 'SGDClassifier':
            model = SGDClassifier(max_iter=1000,
                                tol=1e-3,
                                loss='hinge',
                                class_weight=None)
        
        else:
            print('The model type was not identified')

        return model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        with open(self.model_name, 'wb') as f:
            pickle.dump(self.model, f)
        
    def predict(self, vector):
        y_pred = self.model.predict(vector)

        return y_pred
        
    
    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        score = f1_score(y, y_pred, average='weighted')
        # score2 = self.model.score(X, y)
        # score2 = accuracy_score(y, y_pred)

        return score

if __name__ == "__main__":
    lda = LDA()
    # X_train, y_train = lda.get_feature_vec(type_data='train')
    # X_test, y_test = lda.get_feature_vec(type_data='test')
    # train_data = {'samples': X_train, 'labels': y_train}
    # test_data = {'samples': X_test, 'labels': y_test}

    # with open('./data/train_data_ver2.pkl', 'wb') as f:
    #     pickle.dump(train_data, f)
    
    # with open('./data/test_data_ver2.pkl', 'wb') as f:
    #     pickle.dump(test_data, f)

    with open('./data/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open('./data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X_train, y_train = train_data['samples'], train_data['labels']
    X_test, y_test = test_data['samples'], test_data['labels']


    classifier = Classifier(type_model='SVM')
    print('training SVM classifier')
    classifier.train(X_train, y_train)
    score = classifier.evaluate(X_test, y_test)
    print('SVM', score)

    classifier = Classifier(type_model='LogisticRegression')
    print('training LogisticRegression classifier')
    classifier.train(X_train, y_train)
    score = classifier.evaluate(X_test, y_test)
    print('LogisticRegression', score)

    classifier = Classifier(type_model='SGDClassifier')
    print('training SGDClassifier classifier')
    classifier.train(X_train, y_train)
    score = classifier.evaluate(X_test, y_test)
    print('SGDClassifier', score)