import numpy as np
import pandas as pd

def convert_to_gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def normalize(X):
    X = X.astype(int)
    X = (X-128)/128
    return X

def preprocess(X):
    X = convert_to_gray(X)
    X = normalize(X)
    X = np.expand_dims(X, axis=3)
    return X

signs = pd.read_csv("data/signnames.csv")
def get_sign_name(class_id):
    return signs[signs['ClassId'] == class_id]['SignName'].tolist()[0]