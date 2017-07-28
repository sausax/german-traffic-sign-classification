import numpy as np
import pandas as pd

def convert_to_gray(X):
    wt_arr = [0.299, 0.587, 0.114]
    X = np.einsum("ijkl,l -> ijk", X, wt_arr)
    X = X[:, :, :, np.newaxis]
    return X

def normalize(X):
    X = X.astype(int)
    X = (X-128)/128
    return X

def preprocess(X):
    X = convert_to_gray(X)
    X = normalize(X)
    return X

signs = pd.read_csv("data/signnames.csv")
def get_sign_name(class_id):
    return signs[signs['ClassId'] == class_id]['SignName'].tolist()[0]