from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import gensim
import pickle
# import pickle

import os

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'data')


def get_data(folder_path, mode=None):
    type_data = folder_path.split('/')[-1].split('_')[0].lower()

    if mode is None:
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
    
    elif mode == 'from_file':
        with open('./data/X_' + type_data + '.pkl', 'rb') as f:
            X = pickle.load(f)

        with open('./data/y_' + type_data + '.pkl', 'rb') as f:
            y = pickle.load(f)

    return X, y

def pre_process_doc(path_doc):
    with open(path_doc, 'r', encoding='utf-16') as f:
        lines = f.readlines()
        lines = ' '.join(lines)
        lines = ViTokenizer.tokenize(lines)
        lines = gensim.utils.simple_preprocess(lines)  # remove symbols
        lines = ' '.join(lines)
    
    return [lines]
