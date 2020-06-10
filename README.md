# Representation-and-Distance-Metric-Learning
We present an exploration into distance metric learning, by considering the retrieval process involved in Person Re-Identification (Re-Id). Re-Id is the task of correctly identifying images of the same individual taken from disjoint camera views (i.e CCTV camera network), or from the same camera on different occasions. Our objective is to improve on baseline methods that perform K-Nearest Neighbours (K-NN) retrieval experiments, adhering to standard practices in pattern recognition. We endeavour to find an optimal approach that minimizes the retrieval error from computed ranklists (i.e @Rank1,@Rank10), whilst taking into consideration computational complexity.
## Training/Testing Instructions
• PREREQUISITES: Ensure you have an installed python environment on your machine such as the latest Python or Anaconda distribution. Also ensure you have the following dependencies/libraries installed:
1. Tensorflow
2. Keras
3. SciPy
4. NumPy
5. Pandas
6. matplotlib
7. sklearn
8. metric-learn

• The code for each model used in this report is self
contained within its own single Python Script, in other
words it trains, tests and outputs the accuracy of a model
in the form of graphs and/or terminal outputs.

• If you wish to verify the accuracy of a given model, then
proceed by following these instructions:
1. Download the zip file of our
GitHub code repository at this link:
https://github.com/RajanPatel97/Representation-and-Distance-Metric-Learning/archive/master.zip

2. Unzip the zipped folder and extract its contents
into its own new folder on your machine. In this
folder you will see two folders: Python Scripts and
Jupyter Notebooks.

3. Copy the contents of the CUHK03-NP dataset provided for this coursework into the
Python Scripts folder. It is essential that the
cuhk03 new protocol config labeled.mat and feature data.json files are in the Python Scripts folder,
otherwise the respective python scripts of each
model will not be able to load the data-set and will
fail to run in its entirety.

4. You can now run any Python Script for any of
the models provided in the Python Scripts folder,
which will generate all the relevant results reported.

• It is important to note that some files may take a long
time to train and test, sometimes over 24 hours depending on the computational power you have at your disposal. All of our models were trained and tested using
a Nvidia 840M GPU (lower-end laptop GPU)

• If you wish to run a given model in an interactive
Jupyter notebook format then simply add the contents
of the CUHK03-NP data-set we were provided with and
put it inside the Jupyter Notebooks Folder. Then simply
open a Jupyter Notebook and run the code cell by cell.
Each model has the same name in both .py and .ipynb
formats.

All the files used for training and testing can be found on our GitHub repository: https://github.com/RajanPatel97/Representation-and-Distance-Metric-Learning
