class LoadModel:
    def __init__(self, corpus, model, y_train):
        self.corpus = corpus
        self.model = model
        self.y = y_train
        self.train_vecs = []

    def get_train_vecs(self):
        for i in range(len(self.corpus)):
            top_topics = self.model.get_document_topics(self.corpus[i], minimum_probability=0.0)
            label = LoadCorpusLabel(self.y).get_label()
            self.train_vecs.append([top_topics, label])
        return self.train_vecs


class LoadCorpusLabel:
    def __init__(self, y_train):
        self.y = y_train

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

    def get_label(self):
        label = []
        for i in range(len(self.y)):
            label.append(self.convert_to_onehot(self.y[i]))
        return label
