
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


np.random.seed(42)


# ## Load Data

# In[2]:




training_file = 'data/train.p'
valid_file = 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(valid_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ## Summary of dataset

# In[3]:

# Number of training example
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = 43

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ## Image of each traffic sign

# In[4]:

label_indx_map = {}

for idx, label in enumerate(y_train):
    if label not in label_indx_map:
        label_indx_map[label] = idx
        
### Printing example of each kind
import pandas as pd

signs = pd.read_csv("data/signnames.csv")
def get_sign_name(class_id):
    return signs[signs['ClassId'] == class_id]['SignName'].tolist()[0]

plt.figure(figsize=(20,30))
for idx, key in enumerate(label_indx_map):
    sign_name = get_sign_name(key)
    plt.subplot(15, 3, idx+1)
    plt.imshow(X_train[label_indx_map[key]])
    plt.title(sign_name)
    plt.axis('off')
plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=1.0)


# ## Preprocess images

# In[5]:

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

X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test = preprocess(X_test)


# In[6]:

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)


# ## Model parameters

# In[7]:

batch_size = 128
num_classes = 43
epochs = 20
input_shape = (32, 32, 1)


# In[8]:

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ## LeNet Model

# In[9]:

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# ## Train Classifier

# In[10]:

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_valid, y_valid))


# In[ ]:

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('trained_model.h5')
