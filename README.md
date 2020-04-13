# Virus2Drug
A Drug2vec repository for the drug prediction of COVID-19
## 1.project information 
1. @ARTICLE{SG-LSTM-FRAME,
2. author = {Weidun Xie(shelton xie),Jiawei Luo,Chu Pan,Ying Liu},  
3. title = {Virus2Drug:an embedding based virus-drug recommend system for COVID-19 virus},
4. year = {2020},  
5. journal = {},}  

## 2.Environment
[tab][tab]Hardware: Core: Intel i7-7700HQ ; Graphic Card：GTX1060 6G; RAM:32 GB ddr4 2666,
[tab][tab]CUDA Version 10.0.130,
CuDNN version 7.4.2,
OS: Win 10 64bit,
Python version: IDLE (Python 3.7 64-bit)#yep I wrote directly in IDLE :),

## 3.How to run it?
Basically, you will need to download dataset and retrain the model(from step 1)
or you could directly load the model I trained to repeat the experiments.
1.firstly, you have to unzip “samples_embedding128_default.rar ” and put "samples_embedding128_default.csv" in the folder where the 
"1 train a model.py" existed.
2.secondly, run "1 train a model.py" and wait for the finishing of the training
3.unzip "suspect_drug_pair_dataset.rar" and  run "2 test COVID-samples.py", load the model you trained. What you will get are two files named"LSTM18368_score.npy" and "LSTM_y_label.npy"
4. run "3 convert npy into csv.py" to convert npy file into CSV format file.
5. run "4 sum up the result.py" to the an overview result of prediction of COVID-19 samples
6. run "5 rank_handler.py" to generate the final rank of the samples
