Folder "images_cuhk03" contains all the images from the dataset. Their names may seem illogical at first, but they are properly annotated in the file "cuhk03_new_protocol_config_labeled.mat".


There are 6 main components of this file:
1. camId specidies whether image was taken from camera 1 or camera 2. While evaluating your algorithms, you should not consider images of your current query identity taken from the same camera. For example, when you create ranklist for the first query image (index 22, label 3, camera 1, name "1_003_1_02.png"), you should not include images with indexes 21, 23, 24 in this ranking list
2. filelist
3. gallery_idx, which specifies indexes of the part of the dataset from which you compose your ranklists during testing phase
4. labels contains ground truths for each image
5. query_idx contains indexes of query images
6. train_idx contains indexes of images that can be used for training
To load "cuhk03_new_protocol_config_labeled.mat" into Python you can use the code from this example:
####
train_idxs = loadmat(root_path + "cuhk03_new_protocol_config_labeled.mat")['train_idx'].flatten()
####
with Matlab you can simply use load('cuhk03_new_protocol_config_labeled.mat') or double click on the file in File Explorer.


Apart from two files mentioned above, file "feature_data.json" contains an array 14096x2048 with some feature vectors representing each image in the dataset. With Pyhon you can simply load it as follows:
####
import json
with open("feature_data.json", 'r') as f:
    features = json.load(f)
####

and with Matlab:
####
features = jsondecode(fileread('feature_data.json'));
####



References:
W. Li, R. Zhao, T. Xiao, and X. Wang, “Deepreid: Deep filter pairing neural network for person re-identification,” in 2014 IEEE Conference on Computer Vision and Pattern Recognition, June 2014, pp. 152–159.

Z. Zhong, L. Zheng, D. Cao, and S. Li, “Re-ranking person re-identification with k-reciprocal encoding,” CoRR, vol. abs/1701.08398, 2017. [Online]. Available: http://arxiv.org/abs/1701.08398

CUHK03 dataset:
http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html
https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP

