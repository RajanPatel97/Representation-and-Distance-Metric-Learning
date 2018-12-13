
# coding: utf-8

# In[1]:


import numpy as np
import random
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler


#Store Data Variables
import json
with open('feature_data.json', 'r') as f:
 features = json.load(f)

from scipy.io import loadmat
train_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
query_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
labels = loadmat('cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
gallery_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
filelist = loadmat('cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()
camId = loadmat('cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()



#scaler = StandardScaler()
print(np.array(features))
#features = scaler.fit_transform(features)
X = np.array(features)
y = np.array(labels)
filelist = np.array(filelist)
camId = np.array(camId)


# In[10]:


mask_train = np.array(train_idxs).ravel()
mask_query = np.array(query_idxs).ravel()
mask_gallery = np.array(gallery_idxs).ravel()

mask_train = np.subtract(mask_train, 1)
mask_query = np.subtract(mask_query, 1)
mask_gallery = np.subtract(mask_gallery, 1)


X_train, X_query, X_gallery = X[mask_train, :], X[mask_query, :], X[mask_gallery, :]
y_train, y_query, y_gallery = y[mask_train], y[mask_query], y[mask_gallery]
filelist_train, filelist_query, filelist_gallery = filelist[mask_train], filelist[mask_query], filelist[mask_gallery]
camId_train, camId_query, camId_gallery = camId[mask_train], camId[mask_query], camId[mask_gallery]



import pandas as pd 
import numpy as np

from keras import layers, optimizers, regularizers
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential

from keras.utils import plot_model
#from kt_utils import *
import keras.backend as K

from sklearn import preprocessing, model_selection 
from keras.wrappers.scikit_learn import KerasRegressor


# In[17]:


# create model
model = Sequential()
# layer 1
model.add(Dense(6144, input_dim=6144, activation='relu', kernel_initializer='normal'))
#layer 2
model.add(Dense(1024, activation='relu', kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#layer 3
model.add(Dense(128, activation='relu',kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#layer 4
model.add(Dense(2, activation='softmax'))
# Compile model
from keras import metrics
#optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov= False)
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer, loss='categorical_crossentropy') 


# In[18]:


X_train_pairs = []
y_train_pair_lables = []
for Xtnn, ytnn, camId_t in zip(X_train, y_train, camId_train):
    for Xtnn2, ytnn2, camId_t2 in zip(X_train, y_train, camId_train):
        if ((camId_t == camId_t2) and (ytnn == ytnn2)):
            continue
        elif (ytnn == ytnn2):
            #for i in range (30):
            randindex = random.randint(0, 6401)
            Xtnn3 = X_train[randindex]
            ytnn3 = y_train[randindex]                 

            dist2 = 0
            if(ytnn == ytnn3):
                dist3 = 0
            else:
                dist3 = 1

            Xconcat = np.concatenate((Xtnn,Xtnn2,Xtnn3), axis = None)
            X_train_pairs.append(Xconcat)
            y_train_pair_lables.append((dist2,dist3))   

            Xconcat = np.concatenate((Xtnn,Xtnn3,Xtnn2), axis = None)
            X_train_pairs.append(Xconcat)
            y_train_pair_lables.append((dist3,dist2)) 

y_train_pair_lables = np.array(y_train_pair_lables)
X_train_pairs = np.array(X_train_pairs)
model.fit(X_train_pairs, y_train_pair_lables, batch_size = 150, epochs=20)


# In[19]:


def get_acc_score(y_valid, y_q, tot_label_occur):
    recall = 0
    true_positives = 0
    
    k = 0
    
    max_rank = 30
    
    rank_A = np.zeros(max_rank)
    AP_arr = np.zeros(11)
    
    while (recall < 1) or (k < max_rank):
        if (y_valid[k] == y_q):
            
            true_positives = true_positives + 1
            recall = true_positives/tot_label_occur
            precision = true_positives/(k+1)
            
            AP_arr[round((recall-0.05)*10)] = precision
            
            for n in range (k, max_rank):
                rank_A[n] = 1
            
        k = k+1
        
    max_precision = 0
    for i in range(10, -1, -1):
        max_precision = max(max_precision, AP_arr[i])
        AP_arr[i] = max_precision
    
    AP_ = AP_arr.sum()/11
    
    return AP_, rank_A


# In[ ]:


rank_accuracies = []
AP = []
for val, camId_v, y_v  in zip(X_query, camId_query, y_query):
    v_v_dists = []
    y_valid = []
    for val2, camId_v2, y_v2  in zip(X_gallery, camId_gallery, y_gallery):
        if ((camId_v == camId_v2) and (y_v == y_v2)):
            continue
        else:
            randindex = random.randint(0, 772)
            val3 = X_gallery[randindex]
            y_v3 = y_gallery[randindex]

            dist = model.predict(np.concatenate((val, val2, val3)).reshape((1,6144)))[0][0]
            v_v_dists.append(dist)
            y_valid.append(y_v2)

    tot_label_occur = y_valid.count(y_v)
    v_v_dists = np.array(v_v_dists)
    y_valid = np.array(y_valid)

    
    _indexes = np.argsort(v_v_dists)

    # Sorted distances and labels
    v_v_dists, y_valid = v_v_dists[_indexes], y_valid[_indexes]
    print(v_v_dists)
    
    if tot_label_occur != 0:
        AP_, rank_A = get_acc_score(y_valid, y_v, tot_label_occur)

        AP.append(AP_)

        rank_accuracies.append(rank_A) 

    #if q  > 5:
    #    break
    #q = q+1

rank_accuracies = np.array(rank_accuracies)

total = rank_accuracies.shape[0]
rank_accuracies = rank_accuracies.sum(axis = 0)
rank_accuracies = np.divide(rank_accuracies, total)

i = 0
print ('Accuracies by Rank:')
while i < rank_accuracies.shape[0]:
    print('Rank ', i+1, ' = %.2f%%' % (rank_accuracies[i] * 100), '\t',
          'Rank ', i+2, ' = %.2f%%' % (rank_accuracies[i+1] * 100), '\t',
          'Rank ', i+3, ' = %.2f%%' % (rank_accuracies[i+2] * 100), '\t',
          'Rank ', i+4, ' = %.2f%%' % (rank_accuracies[i+3] * 100), '\t',
          'Rank ', i+5, ' = %.2f%%' % (rank_accuracies[i+4] * 100))
    i = i+5

AP = np.array(AP)

mAP = AP.sum()/AP.shape[0]
print('mAP = %.2f%%' % (mAP * 100))

