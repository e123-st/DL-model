In our work, the transformer-based language models were trained and tested with the code in 'Transformer_Encoder.py' under python 3.8 environment. There are three types of models, which are named 'Bert', 'BertforRegression', and 'BertforRegreesionPlusMLP'.
  
__Bert__  
'Bert' is a model type for pre-trained, which can be trained using masked language modeling method. Refering to the 'Train_Bert.py'  in /example/training/,  the transformer-encoder model could be pre-trained.
  
__BertforRegression__  
'BertforRegression' is a model type for fine-tuning. Refering to the 'Train_BertforRegression.py'  in /example/training/,  the model could be trained.
To validate the predition peformance of  model, the 'Eval_BertforRegression.py'  in /example/validation/ should be used.
To only predict the result via model, the 'Pred_BertforRegression.py'  in /example/prediction/ should be used.
  
__BertforRegressionPlusMLP__   
'BertforRegressionPlusMLP' is a model type for fine-tuning. Refering to the 'Train_BertforRegressionPlusMLP.py'  in /example/training/,  the model could be trained.
To validate the predition peformance of  model, the 'Eval_BertforRegressionPlusMLP.py'  in /example/validation/ could be used.
To only predict the result via model, the 'Pred_BertforRegressionPlusMLP.py'  in /example/prediction/ could be used.
  
__DL models__   
The weights of models developed in our work can be found in DL_model.

There are different kinds of input files in /example/input_file/, which are both the original Excel tables or txt files and splited txt files. The files of training/validation/test sets can be splited with the code in 'DataSplit.py'. Besides, the 'DataSplit.py' can also move the MOFids, labels,and continuous datas from Excel table to corresponding file instead of spliting data into training/validation/test sets.
