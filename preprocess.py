from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import gensim
# import pickle

import os

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'data')


def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    print(dirs)
    for path in dirs:
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths, desc=path):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding='utf-16') as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)
                lines = gensim.utils.simple_preprocess(lines)  # remove symbols
                lines = ' '.join(lines)

                X.append(lines)
                y.append(path)

    return X, y

#
# test_path = os.path.join(dir_path, 'test_full')
# train_path = os.path.join(dir_path, 'train_full')
# processed_path = os.path.join(dir_path, 'data_processed')
#
# # Test data
# X_test, y_test = get_data(test_path)
# pickle.dump(X_test, open(os.path.join(processed_path, 'X_test.pkl'), 'wb'))
# X_test.clear()
# pickle.dump(y_test, open(os.path.join(processed_path, 'y_test.pkl'), 'wb'))
# y_test.clear()
#
# # Train data
# X_train, y_train = get_data(train_path)
# pickle.dump(X_train, open(os.path.join(processed_path, 'X_train.pkl'), 'wb'))
# X_train.clear()
# pickle.dump(y_train, open(os.path.join(processed_path, 'y_train.pkl'), 'wb'))
# y_train.clear()
