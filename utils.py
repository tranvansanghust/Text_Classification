def remove_stopwords(texts, stopwords):
    texts_processed = []
    for doc in texts:
        spacy_doc = []
        for word in doc.split(' '):
            if word not in stopwords:
                spacy_doc.append(word)
        texts_processed.append(spacy_doc)
    return texts_processed

def load_stopwords():
    l = []
    f = open('./data/vietnamese-stopwords.txt', 'r')
    word = f.readline()

    while word:
        l.append(word.replace('\n', '').replace(' ', '_'))
        word = f.readline()
    
    f.close()
    return l