
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




# In[9]:


#scaler = StandardScaler()

#features = scaler.fit_transform(features)
X = np.array(features)
print(X)
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



X_val = []
y_val = []
camId_val = []
val_ind = []
for i in range(7368):
        if(i not in val_ind):
            X_val.append(X_train[i])
            y_val.append(y_train[i])
            camId_val.append(camId_train[i])
            val_ind.append(i)
            for j in range(7368):
                if(y_train[i] == y_train[j] and i != j):
                    X_val.append(X_train[j])
                    y_val.append(y_train[j])
                    camId_val.append(camId_train[j])
                    val_ind.append(j)
            if ((len(set(y_val)) > 99)):
                break


# In[17]:


X_val = np.array(X_val)
y_val = np.array(y_val)
camId_val = np.array(camId_val)




# In[22]:


X_train_new = []
y_train_new = []
camId_train_new = []
for i in range(7368):
    if(i not in val_ind):
        X_train_new.append(X_train[i])
        y_train_new.append(y_train[i])
        camId_train_new.append(camId_train[i])


# In[23]:


X_train_new = np.array(X_train_new)
y_train_new = np.array(y_train_new)
camId_train_new = np.array(camId_train_new)



# In[27]:


mask_vquery = np.random.choice(np.arange(966), 193, replace=False)
mask_vgallery = np.array(list(set(np.arange(0, 966)) - set(mask_vquery)))

X_vquery, X_vgallery =  X_val[mask_vquery, :], X_val[mask_vgallery, :]
y_vquery, y_vgallery =  y_val[mask_vquery], y_val[mask_vgallery]
camId_vquery, camId_vgallery = camId_val[mask_vquery], camId_val[mask_vgallery]


# In[29]:


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


# In[30]:


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


# In[31]:


X_train_pairs = []
y_train_pair_lables = []
for Xtnn, ytnn, camId_t in zip(X_train_new, y_train_new, camId_train_new):
    for Xtnn2, ytnn2, camId_t2 in zip(X_train_new, y_train_new, camId_train_new):
        if ((camId_t == camId_t2) and (ytnn == ytnn2)):
            continue
        elif (ytnn == ytnn2):
            #for i in range (30):
            randindex = random.randint(0, 6401)
            Xtnn3 = X_train_new[randindex]
            ytnn3 = y_train_new[randindex]                 

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


# In[32]:


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


# In[33]:


rank_accuracies = []
AP = []

for val, camId_v, y_v  in zip(X_vquery, camId_vquery, y_vquery):
    v_v_dists = []
    y_valid = []
    for val2, camId_v2, y_v2  in zip(X_vgallery, camId_vgallery, y_vgallery):
        if ((camId_v == camId_v2) and (y_v == y_v2)):
            continue
        else:
            randindex = random.randint(0, 772)
            val3 = X_vgallery[randindex]
            y_v3 = y_vgallery[randindex]

            dist = model.predict(np.concatenate((val, val2, val3)).reshape((1,6144)))[0][0]
            #print(dist)
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


# In[40]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(8.0, 6.0))
color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold', 'lightgreen']
plt.plot(np.arange(1, 31),[47.67,60.10,70.47,75.65,80.31,83.42,86.53,87.56,88.60,88.60,89.64,90.67,91.19,92.7,92.75,93.26,93.78,93.78,93.78,93.78,93.78,93.78,94.30,94.82,94.82,94.82,94.82,94.82,95.3,95.34], color='darkorange', linestyle=':', label='SGD')
plt.plot(np.arange(1, 31),[53.37,70.47,77.72,82.90,84.97,87.56,89.12,90.67,91.71,93.26,93.78,95.34,95.85,96.89,96.89,97.41,97.41,97.41,97.41,97.93,97.93,97.93,97.93,97.93,97.93,97.93,97.93,97.93,97.93, 98.45], color='cyan', linestyle=':', label='Adam')
plt.title('CMC Curves for different optimizers')
plt.xlabel('Rank')
plt.ylabel('Recogniton Accuracy / %')
plt.legend(loc='best')


# In[45]:


plt.figure(figsize=(8.0, 6.0))
color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold', 'lightgreen']
plt.plot(np.arange(1, 31),[53.37,70.47,77.72,82.90,84.97,87.56,89.12,90.67,91.71,93.26,93.78,95.34,95.85,96.89,96.89,97.41,97.41,97.41,97.41,97.93,97.93,97.93,97.93,97.93,97.93,97.93,97.93,97.93,97.93,98.45], color='cyan', linestyle=':', label='20 Epochs')
plt.plot(np.arange(1, 31),[50.26,64.40,71.73,76.96,79.58,84.82,86.39,87.96,88.48,90.05,92.15,92.15,92.15,92.15,92.15,92.67,92.67,93.19,93.72,93.72,94.24,94.76,94.76,95.29,95.29,95.29,95.29,95.29,95.29,95.29], color='magenta', linestyle=':', label='15 Epochs')
plt.plot(np.arange(1, 31),[48.19,59.59,69.43,75.65,78.24,82.38,83.94,85.49,85.49,87.05,87.56,88.60,89.12,90.16,91.19,91.71,92.23,92.75,92.75,93.26,93.26,93.78,93.78,94.30,94.82,94.82,94.82,94.82,94.82,94.82], color='orange', linestyle=':', label='10 Epochs')
plt.plot(np.arange(1, 31),[34.90,46.88,58.33,68.75,72.92,76.04,77.08,79.17,79.69,81.77,82.81,82.81,84.38,86.46,86.98,88.02,88.02,88.02,88.02,88.54,89.06,89.58,90.62,91.67,91.67,93.23,93.75,93.75,94.27,94.27], color='green', linestyle=':', label='5 Epochs')
plt.title('CMC Curves for varying training epochs')
plt.xlabel('Rank')
plt.ylabel('Recogniton Accuracy / %')
plt.legend(loc='best')


# In[46]:


plt.figure(figsize=(8.0, 6.0))
color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold', 'lightgreen']
plt.plot(np.arange(1, 31),[47.15,61.66,68.91,75.65,77.20,79.27,82.38,84.97,85.49,87.05,88.08,88.60,90.67,91.19,92.75,92.75,93.26,93.26,94.30,94.82,94.82,95.34,95.34,95.85,95.85,95.85,96.37,96.37,96.37,96.37], color='red', linestyle=':', label='batch size = 200')
plt.plot(np.arange(1, 31),[47.67,60.10,68.91,76.17,79.27,80.83,82.90,85.49,86.01,87.05,88.60,89.64,90.67,91.71,92.23,92.75,92.75,92.75,92.75,92.75,92.75,93.26,93.78,93.78,95.34,95.34,95.85,96.37,96.37,96.37], color='cyan', linestyle=':', label='batch size = 175')
plt.plot(np.arange(1, 31),[49.74,63.21,70.98,78.24,81.87,84.97,87.56,90.16,91.71,94.30,94.82,94.82,95.85,97.41,97.41,97.41,97.41,97.93,97.93,97.93,97.93,97.93,97.93,97.93,98.45,98.45,98.45,98.45,98.96,98.96], color='magenta', linestyle=':', label='batch size = 150')
plt.plot(np.arange(1, 31),[47.12,63.87,70.68,74.87,76.44,79.58,81.15,83.77,85.86,86.91,87.96,88.48,89.53,89.53,90.58,91.62,92.67,93.72,93.72,94.76,94.76,95.81,95.81,95.81,95.81,95.81,95.81,96.34,96.34,96.34], color='orange', linestyle=':', label='batch size = 125')
plt.plot(np.arange(1, 31),[34.90,46.88,58.33,68.75,72.92,76.04,77.08,79.17,79.69,81.77,82.81,82.81,84.38,86.46,86.98,88.02,88.02,88.02,88.02,88.54,89.06,89.58,90.62,91.67,91.67,93.23,93.75,93.75,94.27,94.27], color='green', linestyle=':', label='batch size = 100')
plt.title('CMC Curves for varying batch sizes')
plt.xlabel('Rank')
plt.ylabel('Recogniton Accuracy / %')
plt.legend(loc='best')

