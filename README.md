# EEG-classification
You can find the code in **lib** that analysis the EEG data by different methods, including MLP, LSTM and CNN. Before running the code, you need to download the dataset from [here](http://perceive.dieei.unict.it/index-dataset.php?name=EEG_Data). To run the code, do
```
python main.py -ed PATH/TO/EEG/DATASET -sp PATH/TO/EEG/SPLITS -c CLASSIFIER_NAME 
```