import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import DataCollatorForLanguageModeling, DefaultDataCollator 
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, Trainer 
from tokenizer.DatasetWithTokenizer import MOFid_Tokenizer
from tokenizer.DatasetWithTokenizer import token_encode
from tokenizer.DatasetWithTokenizer import token_encode_with_MLPinputs
from tokenizer.DatasetWithTokenizer import token_encode_with_labels
from tokenizer.DatasetWithTokenizer import token_encode_with_MLPinputs_and_Regresslabels
from MLPmodel.MLPPredictor import BertforRegression
from MLPmodel.MLPPredictor import BertforRegressionPlusMLP
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.manifold import TSNE
import sys
import os
            
np.set_printoptions(threshold=sys.maxsize) 

class Models():
    
     def __init__(self,Model_Type: str,                       
                       vocab_path: str, 
                       output_dir = None,                      
                       max_len = 512,
                       use_GPU = True,  
                       use_Trainer = True,
                       use_Accelerate = False,                                              
                       **kwargs):

         """                          
         Model_Type: 'Bert': pre-trained model， only be used in train function
         
                     'BertforRegression': fintuning model (regression),
                                          input: mofid, 
                                          can be used in train eval, and predict function
                                          
                     'BertforRegressionPlusMLP': fintuning model (regression), 
                                                 input: mofid and continuous data,
                                                 can be used in train eval, and predict function
                     
            
         output_dir: the output directory where the model predictions and checkpoints will be written
                     if use_Trainer = True，it should be set, end with '/', str
                                 
         vocab_path: the directory of vocabluary, str            

         max_len: max length of input sentences excluding [CLS] and [SEP]，default to 512, int  
                  
         use_GPU: whether to use GPU
                  if True, the GPU will be used, if it exist.
                  if False, only the CPU will be used.                            
                  default to True
                  bool
                 
         use_Trainer: whether to use the Trainer developed by huggingface
                      if True，the 'Accelerate' will be used
                      for train function, only True can be set.
                      for eval and predict, both True and False can be selected.
                      default to True
                      bool
           
                               
         use_Accelerate: whether to use the 'Accelerate' 
                         if use_Trainer = True，the 'Accelerate' will be used
                         default to False
                         bool 
                        
         """
         
         self.Model_Type = Model_Type
         self.output_dir = output_dir
         self.vocab_path = vocab_path
         self.max_len = max_len
         self.max_position_embeddings = self.max_len + 2
         self.use_GPU = use_GPU
         self.use_Trainer = use_Trainer
         self.use_Accelerate = use_Accelerate
         
     def train(self,num_hidden_layers = 12,
                    num_attention_heads = 12,
                    MLP_input_neurons = None,
                    MLP_output_neurons = None,
                    MLP_num_hidden_layer = None,
                    MLP_num_hidden_neurons = None,
                    MLP_activation = None,
                    Head_input_neurons = None,
                    Head_output_neurons = None,
                    Head_num_hidden_layer = None,
                    Head_num_hidden_neurons = None,
                    Head_activation = None,
                    loss_weights = None,
                    train_batch_size = 8, 
                    eval_batch_size = 8,
                    learning_rate = 1e-4,
                    num_train_epochs = 100,
                    weight_decay = 1e-2,
                    warmup_ratio =  0.05,
                    evaluation_strategy = 'steps',
                    save_strategy = 'steps',
                    save_steps = 500,
                    eval_steps = 500,
                    save_safetensors = False,
                    load_best_model_at_end = True,
                    metric_for_best_model = 'eval_loss',
                    greater_is_better = False,
                    BertPretrainModel_path = None,
                    model_save_path = None,                        
                    train_mofid_path = None,
                    eval_mofid_path = None,
                    train_gf_path = None,
                    eval_gf_path = None,
                    train_labels_path = None,
                    eval_labels_path = None,
                    vocab_size = 30522, 
                    hidden_size = 768,
                    intermediate_size = 3072, 
                    hidden_act = "gelu", 
                    hidden_dropout_prob = 0.1, 
                    attention_probs_dropout_prob = 0.1,
                    type_vocab_size = 1, 
                    initializer_range = 0.02, 
                    layer_norm_eps = 1e-12, 
                    pad_token_id = 0, #填充标记
                    bos_token_id = 2, #开始标记
                    eos_token_id = 3, #结束标记
                    position_embedding_type = "absolute", 
                    use_cache = True, 
                    classifier_dropout = None,                    
                    num_labels = 1,
                    output_hidden_states = True,
                    output_attentions = True, 
                    KeepTraining = False):

                          
         """
         num_hidden_layers: number of transformer layers，int                            
                            
         num_attention_heads: number of attention heads，int
                                                        
         MLP_X: hyperparamerter for MLP model, int
         Head_X: hyperparamerter for MLP head, int
         
         X:
         num_hidden_layer：number of hidden layer, int or None
         
         num_hidden_neurons: number of neurons in each hiddden layers, list,like: [1,2,3,4] or None
         
         activation： activation function, default to 'relu'
                     both relu, tanh, and sigmoid are selectable.
                  
         loss_weight: the weight for loss of different task，list, only for multi-task model
                  
         train_batch_size：the batch size for training, int 
         
         eval_batch_size: the batch size for validation, int     
         
         learning_rate： the initial learning rate for AdamW optimizer, float
         
         num_train_epochs: total number of training epochs， int
                                
         vocab_size: the size of vocabluary, int
         
         hidden_size: the fiddeen size in transformer encoder,int   
         
         pad_token_id: the id of [PAD], int
         
         bos_token_id: the id of [CLS], int
         
         eos_token_id: the id of [SEP], int
                             
         output_hidden_states: whether to output the hidden states matrix, default to True, bool
         
         output_attentions: whethe rto output the attention heads, default to True, bool
                  
         BertPretrainModel_path:  the directory of pre-trained model, str
         
         model_save_path: the output directory of model, str
         
         train_mofid_path: the directory of MOFid for model training, str
         
         eval_mofid_path: the directory of MOFid for model validation, str
         
         train_gf_path： the directory of continuous data for model training, str
         
         eval_gf_path: the directory of continuous data for model validation, str
         
         train_labels_path: the directory of labels for model training, str
         
         eval_labels_path: the directory of labels for model validation, str
                                                                
         num_labels: number of prediction labels，default to 1，int
                     if Model_Type = 'BertforRegression', 'BertforRegressionPlusMLP' .
          
                                         
         
         KeepTraining: whether to keep training，default to False, bool
         
         Pleas refer to the document of HuggingFace for the following hyperparameters.
         
         evaluation_strategy
         save_step
         eval_step
         weight_decay
         warm_ratio
         intermediate_size
         hidden_act
         hidden_dropout_prob
         attention_probs_dropout_prob
         type_vocab_size
         initializer_range 
         layer_norm_eps
         position_embedding_type  
         use_cache 
         classifier_dropout 
         save_safetensors 
         load_best_model_at_end
         metric_for_best_model
         greater_is_better
         
         """
                  
          
         tokenizer = MOFid_Tokenizer(vocab_file = self.vocab_path)
         

         training_args = TrainingArguments(output_dir = self.output_dir,
                                           evaluation_strategy = evaluation_strategy, 
                                           per_device_train_batch_size = train_batch_size,
                                           per_device_eval_batch_size = eval_batch_size,
                                           save_strategy = save_strategy,
                                           learning_rate = learning_rate,
                                           weight_decay = weight_decay,
                                           num_train_epochs = num_train_epochs,
                                           warmup_ratio = warmup_ratio,
                                           save_steps = save_steps,
                                           save_safetensors = save_safetensors,
                                           eval_steps = eval_steps,
                                           load_best_model_at_end = load_best_model_at_end,
                                           metric_for_best_model = metric_for_best_model,
                                           greater_is_better = greater_is_better)             
         
         if self.Model_Type == 'Bert' :
#dataset                 
            train_dataset = token_encode(data_path = train_mofid_path,
                                         vocab_path = self.vocab_path,
                                         max_length = self.max_position_embeddings)


            eval_dataset = token_encode(data_path = eval_mofid_path,
                                        vocab_path = self.vocab_path,
                                        max_length = self.max_position_embeddings)
            
#data_collator               
            data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer,
                                                            mlm = True,
                                                            mlm_probability = 0.15)   

               
            configuration = BertConfig(vocab_size = vocab_size, 
                                       hidden_size = hidden_size,
                                       num_hidden_layers = num_hidden_layers, 
                                       num_attention_heads = num_attention_heads, 
                                       intermediate_size = intermediate_size, 
                                       hidden_act = hidden_act, 
                                       hidden_dropout_prob = hidden_dropout_prob, 
                                       attention_probs_dropout_prob = attention_probs_dropout_prob,
                                       max_position_embeddings = self.max_position_embeddings,
                                       type_vocab_size = type_vocab_size, 
                                       initializer_range = initializer_range, 
                                       layer_norm_eps = layer_norm_eps, 
                                       pad_token_id = pad_token_id,  #填充标记
                                       bos_token_id = bos_token_id,  #开始标记
                                       eos_token_id = eos_token_id,  #结束标记
                                       position_embedding_type = position_embedding_type, 
                                       use_cache = True, 
                                       classifier_dropout = None)


            model = BertForMaskedLM(configuration)
            
         if self.Model_Type == 'BertforRegression' or self.Model_Type == 'BertforRegressionPlusMLP' :   
            
#data_collator                
            data_collator = DefaultDataCollator()
             
#dataset        
            if self.Model_Type == 'BertforRegression' :
                
               train_dataset = token_encode_with_labels(data_x_path = train_mofid_path,
                                                        data_y_path = train_labels_path,
                                                        vocab_path = self.vocab_path,
                                                        max_length = self.max_position_embeddings)


               eval_dataset = token_encode_with_labels(data_x_path = eval_mofid_path,
                                                       data_y_path = eval_labels_path,
                                                       vocab_path = self.vocab_path,
                                                       max_length = self.max_position_embeddings) 
               

               model = BertforRegression.from_pretrained(BertPretrainModel_path,
                                                         output_hidden_states = output_hidden_states, 
                                                         output_attentions = output_attentions,
                                                         num_labels = num_labels,
                                                         RH_input_neurons = Head_input_neurons,
                                                         RH_output_neurons = Head_output_neurons,
                                                         RH_num_hidden_layer = Head_num_hidden_layer,
                                                         RH_num_hidden_neurons = Head_num_hidden_neurons,
                                                         RH_activation = Head_activation,
                                                         loss_weights = loss_weights,
                                                         use_Trainer = self.use_Trainer)
               
               
            if self.Model_Type == 'BertforRegressionPlusMLP' :    

               train_dataset = token_encode_with_MLPinputs_and_Regresslabels(data_gf_path = train_gf_path,
                                                                             data_mofid_path = train_mofid_path,
                                                                             data_y_path = train_labels_path,
                                                                             vocab_path = self.vocab_path,
                                                                             max_length = self.max_position_embeddings)


               eval_dataset = token_encode_with_MLPinputs_and_Regresslabels(data_gf_path = eval_gf_path,
                                                                            data_mofid_path = eval_mofid_path,
                                                                            data_y_path = eval_labels_path,                                       
                                                                            vocab_path = self.vocab_path,
                                                                            max_length = self.max_position_embeddings)
                                       

               model = BertforRegressionPlusMLP.from_pretrained(BertPretrainModel_path,
                                                                output_hidden_states = output_hidden_states, 
                                                                output_attentions = output_attentions,
                                                                num_labels = num_labels,
                                                                MLP_input_neurons = MLP_input_neurons,
                                                                MLP_output_neurons = MLP_output_neurons,
                                                                MLP_num_hidden_layer = MLP_num_hidden_layer,
                                                                MLP_num_hidden_neurons = MLP_num_hidden_neurons,
                                                                MLP_activation = MLP_activation,
                                                                RH_input_neurons = Head_input_neurons,
                                                                RH_output_neurons = Head_output_neurons,
                                                                RH_num_hidden_layer = Head_num_hidden_layer,
                                                                RH_num_hidden_neurons = Head_num_hidden_neurons,
                                                                RH_activation = Head_activation,                                                                
                                                                use_Trainer = self.use_Trainer)

                       
#train         
         if self.use_Trainer == False :
            print('----------------------')
            print('In this code, the model is trained via the Trainer in Transformers.Please modify the code. ')
         
         
         if self.use_Trainer == True :
            if self.use_GPU == True :
               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
               model = model.to(device)   
               
            if self.use_GPU == False :
               device = torch.device('cpu')
               model = model.to(device)   
                           
            trainer = Trainer(model = model,
                              args = training_args,
                              data_collator = data_collator,
                              train_dataset = train_dataset,
                              eval_dataset = eval_dataset)
            
            if KeepTraining == False :
               trainer.train()
            
            if KeepTraining == True :
               trainer.train(resume_from_checkpoint = True)   

#训练集评估
            train_metrics = trainer.evaluate(train_dataset)

#验证集评估
            eval_metrics = trainer.evaluate(eval_dataset)
 
#存储
            #trainer.save_metrics(split = 'train',metrics = train_metrics)
            #trainer.save_metrics(split = 'eval',metrics = eval_metrics)
            metrics = []
            metrics.append('train : '+ str(train_metrics))
            metrics.append('eval : '+ str(eval_metrics))
            
            output_path=self.output_dir+'metrics.txt'
            output=open(output_path,'w')
            for i in range(len(metrics)):
                output.write(str(metrics[i])+'\n')
            output.close()
                                     
#保存预训练模型
            trainer.save_model(model_save_path)

         return print('Finish.')
     
        
         
     def eval(self, model = None,
                    batch_size = 8,
                    MLP_input_neurons = None,
                    MLP_output_neurons = None,
                    MLP_num_hidden_layer = None,
                    MLP_num_hidden_neurons = None,
                    MLP_activation = None,
                    Head_input_neurons = None,
                    Head_output_neurons = None,
                    Head_num_hidden_layer = None,
                    Head_num_hidden_neurons = None,
                    Head_activation = None,
                    gf_path = None,
                    mofid_path = None,
                    label_path = None,
                    output_path = None,                          
                    split = 'Train',
                    output_hidden_states = True,
                    output_attentions = True, 
                    num_labels = 1,
                    num_loader_workers = 0,
                    output_representation_to_txt = False,
                    output_attentions_to_txt = False,
                    output_tsne_data_to_txt = False):     
         
         """
         model:the directory of model, str
         
         gf_path: the directory of continous data, txt file, str
         
         mofid_path: the directory of MOFid，txt file, str
         
         label_path: the directory of original labels, txt file, str
         
         output_path: the output directory of predction and metric, str, end with '/'
                                                       
         num_loader_workers: number of worker process in dataloader, default to 0, int
                  
         output_representation_to_txt: whether to output the representation vector to a txt file
                                       it works, if use_Trainer = False
                                       default to False
                                       bool
                                
         output_attentions_to_txt: whether to output the attention head to a txt file
                                   it works, if use_Trainer = False
                                   if True, the evaluation metrics will not be calculated.
                                   default to False
                                   bool
                                   
         output_tsne_data_to_txt:  whether to calculate the t-SNE result to a txt file
                                   it works, if use_Trainer = False
                                   default to False
                                   bool                          
                                   
         """
         
         output_path_exist = os.path.exists(output_path) 
         if output_path_exist == False :
            os.mkdir(output_path)
                      
 
         if self.use_GPU == True:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
         if self.use_GPU == False:   
            device = torch.device('cpu')


         if self.Model_Type == 'BertforRegression' :
             
            dataset = token_encode_with_labels(data_x_path = mofid_path,
                                               data_y_path = label_path,
                                               vocab_path = self.vocab_path,
                                               max_length = self.max_position_embeddings)
            
            loader = DataLoader(dataset = dataset,batch_size = batch_size,num_workers = num_loader_workers)
            
            Pred_model = BertforRegression.from_pretrained(model,
                                                           output_hidden_states = output_hidden_states, 
                                                           output_attentions = output_attentions,
                                                           num_labels = num_labels,
                                                           RH_input_neurons = Head_input_neurons,
                                                           RH_output_neurons = Head_output_neurons,
                                                           RH_num_hidden_layer = Head_num_hidden_layer,
                                                           RH_num_hidden_neurons = Head_num_hidden_neurons,
                                                           RH_activation = Head_activation,
                                                           use_Trainer = self.use_Trainer,
                                                           output_representation_to_txt = output_representation_to_txt,
                                                           map_location = device)
            

         if self.Model_Type == 'BertforRegressionPlusMLP' :
             
            dataset = token_encode_with_MLPinputs_and_Regresslabels(data_gf_path = gf_path,
                                                                    data_mofid_path = mofid_path,
                                                                    data_y_path = label_path,
                                                                    vocab_path = self.vocab_path,
                                                                    max_length = self.max_position_embeddings)
        
            loader = DataLoader(dataset = dataset,batch_size = batch_size,num_workers = num_loader_workers)

            Pred_model = BertforRegressionPlusMLP.from_pretrained(model,
                                                                  output_hidden_states = output_hidden_states, 
                                                                  output_attentions = output_attentions,
                                                                  num_labels = num_labels,
                                                                  MLP_input_neurons = MLP_input_neurons,
                                                                  MLP_output_neurons = MLP_output_neurons,
                                                                  MLP_num_hidden_layer = MLP_num_hidden_layer,
                                                                  MLP_num_hidden_neurons = MLP_num_hidden_neurons,
                                                                  MLP_activation = MLP_activation,
                                                                  RH_input_neurons = Head_input_neurons,
                                                                  RH_output_neurons = Head_output_neurons,
                                                                  RH_num_hidden_layer = Head_num_hidden_layer,
                                                                  RH_num_hidden_neurons = Head_num_hidden_neurons,
                                                                  RH_activation = Head_activation,                                                                 
                                                                  use_Trainer = self.use_Trainer,
                                                                  output_representation_to_txt = output_representation_to_txt,
                                                                  map_location = device)



                            
         if self.use_Trainer == True :
             
            print('----------------------')            
            print('Predicted via the Trainer in Transformers. The Accelerate module is used as default. Please pay attention to the memory of your CPUs/GPUs.')
            print('Using the Accelerate module, the tensor amd model will be automatically moved to the same device')
            print('----------------------')

            metrics = []
            
            args = TrainingArguments(output_dir = self.output_dir)
            
            data_collator = DefaultDataCollator()
                            
            trainer=Trainer(model = Pred_model,
                            args = args,
                            data_collator = data_collator)
            
            predict = trainer.predict(dataset)                
            logits = predict[0][0]
            labels = predict[1]
               
 
            
         if self.use_Trainer == False and self.use_Accelerate == False :

            metrics = []                
            labels = []
            logits = []
            
            
            if output_representation_to_txt == True :                
               Representation_output = []
               
            if output_attentions_to_txt == True :
               attention = []
            
            data_loader = iter(loader)
            with tqdm(data_loader) as pbar:
                 pbar.set_description('Processing:')
                 for inputs in data_loader:
                     with torch.no_grad(): 
                          pred = Pred_model(**inputs)
                          labels.append(inputs['labels'])
                          logits.append(pred[0])  
                          if output_representation_to_txt == True :                            
                             Representation_output.append(pred[-1])                               
                          if output_attentions_to_txt == True :
                             attention.append(pred[2])
                          pbar.update(1)


            labels = list(np.ravel(torch.cat(labels).numpy()))
            logits = list(np.ravel(torch.cat(logits).numpy())) 
                
               
            if output_representation_to_txt == True :
               Representation_output = list(torch.cat(Representation_output).numpy())  
               
            if output_attentions_to_txt == True :
               attention = list(attention[0][0].numpy()) 
            
                        
         if self.use_Trainer == False and self.use_Accelerate == True :
             
            print('----------------------')
            print('The Accelerate module is used. Please pay attention to the memory of your CPUs/GPUs.')
            print('Using the Accelerate module, the tensor amd model will be automatically moved to the same device')
            print('----------------------')
            
            from accelerate import Accelerator

            metrics = []                
            labels = []
            logits = [] 
            
            if output_representation_to_txt == True :                
               Representation_output = []
               
            if output_attentions_to_txt == True :
               attention = []
            
          
            accelerator=Accelerator()
                             
            Pred_model, data_loader = accelerator.prepare(Pred_model, loader)
            with tqdm(data_loader) as pbar:
                 pbar.set_description('Processing:')
                 for inputs in data_loader:
                     with torch.no_grad(): 
                          pred = Pred_model(**inputs)
                          labels.append(inputs['labels'])
                          logits.append(pred[0])                          
                          if output_representation_to_txt == True  :
                             Representation_output.append(pred[-1])
                          if output_attentions_to_txt == True :
                             attention.append(pred[2])   
                          pbar.update(1)
                          
            if str(accelerator.device) == 'cuda' :
               labels = list(np.ravel(torch.cat(labels).cpu().numpy()))
               logits = list(np.ravel(torch.cat(logits).cpu().numpy())) 
                  
                  
               if output_representation_to_txt == True :
                  Representation_output = list(torch.cat(Representation_output).cpu().numpy())  
                  
               if output_attentions_to_txt == True :
                  attention = list(torch.cat(attention[0]).cpu().numpy())    
               
            if str(accelerator.device) == 'cpu' :
               labels = list(np.ravel(torch.cat(labels).numpy()))
               logits = list(np.ravel(torch.cat(logits).numpy())) 
                                   
                  
               if output_representation_to_txt == True :
                  Representation_output = list(torch.cat(Representation_output).numpy())  
                  
               if output_attentions_to_txt == True :
                  attention = list(attention[0][0].numpy())    
               

         if output_attentions_to_txt == True :
            token = []
            tokenizer = MOFid_Tokenizer(vocab_file = self.vocab_path)  
            mofid = open(mofid_path,'r').readlines()
            for i in range(len(mofid)):
                token.append(str(tokenizer.tokenize(mofid[i])))


         labels_output_path = output_path+split+'_labels.txt'
         output = open(labels_output_path,'w')
         for i in range(len(labels)):
             output.write(str(labels[i]))
             output.write('\n')
         output.close()

         logits_output_path = output_path+split+'_logits.txt'
         output = open(logits_output_path,'w')
         for i in range(len(logits)):
             output.write(str(logits[i]))
             output.write('\n')
         output.close()
         
         
         if self.Model_Type == 'BertforRegression' and num_labels > 1 :
            Split_Metrics=[]
            for i in range(num_labels):
                label,logit=[],[]
                last=len(labels)-num_labels+i+1  
                for j in np.arange(i,last,num_labels):
                    label.append(labels[j])
                    logit.append(logits[j])
                    
                label_output_path = output_path+split+'_label_'+str(i+1)+'.txt'
                np.savetxt(label_output_path,label,fmt='%f')   
                
                logit_output_path = output_path+split+'_logit_'+str(i+1)+'.txt'
                np.savetxt(logit_output_path,logit,fmt='%f')
                                
                if output_attentions_to_txt == False :
                   r2 = r2_score(label,logit)  
                   mae = mean_absolute_error(label,logit)
                   mse = mean_squared_error(label,logit)
                   Split_Metrics.append(str(i+1)+'  R2 : '+str(r2)+'  MAE : '+str(mae)+'  MSE : '+str(mse)) 
                   
                   Split_Metrics_output_path = output_path+split+'_Split_Metrics.txt'
                   output = open(Split_Metrics_output_path,'w')
                   for i in range(len(Split_Metrics)):
                       output.write(str(Split_Metrics[i]))
                       output.write('\n')
                   output.close()
            
         if output_tsne_data_to_txt == True and output_representation_to_txt == False :  
            print('Error: output_representation_to_txt = False, the tsne data can not be calculated')
         
         if output_tsne_data_to_txt == True and output_representation_to_txt == True :   
            
            print('Warning: the tsne data is being calculated.')
            
            tsne_data=[]            
            tsne_data_output_path = output_path+split+'_tsne_data_output.txt'            
            for i in range(len(Representation_output)):
                tsne_data.append(list(Representation_output[i]))
            
            tsne_data_output = TSNE(n_components=2).fit_transform(np.array(tsne_data))
            
            print('The tsne data has been calculated.')
            
            np.savetxt(tsne_data_output_path,tsne_data_output,fmt='%f') 
             
                                      
         if output_attentions_to_txt == False :
            R2 = r2_score(labels,logits)
            MAE = mean_absolute_error(labels,logits)
            MSE = mean_squared_error(labels,logits)
            metrics.append('R2')
            metrics.append(R2)
            metrics.append('MSE')
            metrics.append(MSE)
            metrics.append('MAE')
            metrics.append(MAE)
                          
            metrics_output_path = output_path+split+'_metrics.txt'
            output = open(metrics_output_path,'w')
            for i in range(len(metrics)):
                output.write(str(metrics[i]))
                output.write('\n')
            output.close()
         
         if output_representation_to_txt == True :
            Representation_output_path = output_path+split+'_Representation_output.txt'
            np.savetxt(Representation_output_path,Representation_output)
            
         if output_attentions_to_txt == True :
            attention_path_exist = os.path.exists(output_path+'Attettion_Head') 
            if attention_path_exist == False :
               os.mkdir(output_path+'Attettion_Head')   
               
            for i in range(len(attention[0])):                 
                  attention_output_path = output_path+'Attettion_Head/'+split+'_attention_Head_'+str(i+1)+'_.txt'
                  np.savetxt(attention_output_path,attention[0][i]) 

            token_output_path = output_path+'Attettion_Head/'+split+'_token.txt'
            output = open(token_output_path,'w')
            for i in range(len(token)):
                output.write(str(token[i]))
                output.write('\n')
            output.close()
            
         return print('Finish.')
     
                     
     def preidct(self, model = None,
                       batch_size = 8,
                       MLP_input_neurons = None,
                       MLP_output_neurons = None,
                       MLP_num_hidden_layer = None,
                       MLP_num_hidden_neurons = None,
                       MLP_activation = None,
                       Head_input_neurons = None,
                       Head_output_neurons = None,
                       Head_num_hidden_layer = None,
                       Head_num_hidden_neurons = None,
                       Head_activation = None,
                       gf_path = None,
                       mofid_path = None,
                       output_path = None,                          
                       output_hidden_states = True,
                       output_attentions = True, 
                       num_labels = 1,
                       num_loader_workers = 0,
                       output_representation_to_txt = False,
                       output_attentions_to_txt = False,
                       output_tsne_data_to_txt = False):     
              
              
              output_path_exist = os.path.exists(output_path) 
              if output_path_exist == False :
                 os.mkdir(output_path)
                           
      
              if self.use_GPU == True:
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 
              if self.use_GPU == False:   
                 device = torch.device('cpu')
     
     
              if self.Model_Type == 'BertforRegression' :
                  
                 dataset = token_encode(data_x_path = mofid_path,
                                        vocab_path = self.vocab_path,
                                        max_length = self.max_position_embeddings)
                 
                 loader = DataLoader(dataset = dataset,batch_size = batch_size,num_workers = num_loader_workers)
                 
                 Pred_model = BertforRegression.from_pretrained(model,
                                                                output_hidden_states = output_hidden_states, 
                                                                output_attentions = output_attentions,
                                                                num_labels = num_labels,
                                                                RH_input_neurons = Head_input_neurons,
                                                                RH_output_neurons = Head_output_neurons,
                                                                RH_num_hidden_layer = Head_num_hidden_layer,
                                                                RH_num_hidden_neurons = Head_num_hidden_neurons,
                                                                RH_activation = Head_activation,
                                                                use_Trainer = self.use_Trainer,
                                                                output_representation_to_txt = output_representation_to_txt,
                                                                map_location = device)
                 
     
              if self.Model_Type == 'BertforRegressionPlusMLP' :
                  
                 dataset = token_encode_with_MLPinputs(data_gf_path = gf_path,
                                                       data_mofid_path = mofid_path,
                                                       vocab_path = self.vocab_path,
                                                       max_length = self.max_position_embeddings)
             
                 loader = DataLoader(dataset = dataset,batch_size = batch_size,num_workers = num_loader_workers)
     
                 Pred_model = BertforRegressionPlusMLP.from_pretrained(model,
                                                                       output_hidden_states = output_hidden_states, 
                                                                       output_attentions = output_attentions,
                                                                       num_labels = num_labels,
                                                                       MLP_input_neurons = MLP_input_neurons,
                                                                       MLP_output_neurons = MLP_output_neurons,
                                                                       MLP_num_hidden_layer = MLP_num_hidden_layer,
                                                                       MLP_num_hidden_neurons = MLP_num_hidden_neurons,
                                                                       MLP_activation = MLP_activation,
                                                                       RH_input_neurons = Head_input_neurons,
                                                                       RH_output_neurons = Head_output_neurons,
                                                                       RH_num_hidden_layer = Head_num_hidden_layer,
                                                                       RH_num_hidden_neurons = Head_num_hidden_neurons,
                                                                       RH_activation = Head_activation,
                                                                       use_Trainer = self.use_Trainer,
                                                                       output_representation_to_txt = output_representation_to_txt,
                                                                       map_location = device)
     
                                       
              if self.use_Trainer == True :
                  
                 print('----------------------')            
                 print('In this modules, the Trainer would not be used.')

     
              if self.use_Trainer == False and self.use_Accelerate == False :
     
                 logits = []
                 
                 
                 if output_representation_to_txt == True :                
                    Representation_output = []
                    
                 if output_attentions_to_txt == True :
                    attention = []
                 
                 data_loader = iter(loader)
                 with tqdm(data_loader) as pbar:
                      pbar.set_description('Processing:')
                      for inputs in data_loader:
                          with torch.no_grad(): 
                               pred = Pred_model(**inputs)                               
                               logits.append(pred[0]) 
                               if output_representation_to_txt == True :
                                  Representation_output.append(pred[-1])
                               if output_attentions_to_txt == True :
                                  attention.append(pred[2])
                               pbar.update(1)
     

                 logits = list(np.ravel(torch.cat(logits).numpy())) 
                    
                    
                 if output_representation_to_txt == True :
                    Representation_output = list(torch.cat(Representation_output).numpy())  
                    
                 if output_attentions_to_txt == True :
                    attention = list(attention[0][0].numpy()) 
                 
                             
              if self.use_Trainer == False and self.use_Accelerate == True :
                  
                 print('----------------------')
                 print('The Accelerate module is used. Please pay attention to the memory of your CPUs/GPUs.')
                 print('Using the Accelerate module, the tensor amd model will be automatically moved to the same device')
                 print('----------------------')
                 
                 from accelerate import Accelerator

                 logits = [] 
                 
                 if output_representation_to_txt == True :                
                    Representation_output = []
                    
                 if output_attentions_to_txt == True :
                    attention = []
                 
            
                 accelerator=Accelerator()
                                  
                 Pred_model, data_loader = accelerator.prepare(Pred_model, loader)
                 with tqdm(data_loader) as pbar:
                      pbar.set_description('Processing:')
                      for inputs in data_loader:
                          with torch.no_grad(): 
                               pred = Pred_model(**inputs)
                               logits.append(pred[0])                          
                               if output_representation_to_txt == True :
                                  Representation_output.append(pred[-1])
                               if output_attentions_to_txt == True :
                                  attention.append(pred[2])   
                               pbar.update(1)
                               
                 if str(accelerator.device) == 'cuda' :

                    logits = list(np.ravel(torch.cat(logits).cpu().numpy())) 
                       
                       
                    if output_representation_to_txt == True :
                       Representation_output= list(torch.cat(Representation_output).cpu().numpy())  
                       
                    if output_attentions_to_txt == True :
                       attention = list(torch.cat(attention[0]).cpu().numpy())    
                    
                 if str(accelerator.device) == 'cpu' :
                    logits = list(np.ravel(torch.cat(logits).numpy())) 
                                         
                       
                    if output_representation_to_txt == True :
                       Representation_output = list(torch.cat(Representation_output).numpy())  
                       
                    if output_attentions_to_txt == True :
                       attention = list(attention[0][0].numpy())    
                    
     
              if output_attentions_to_txt == True :
                 token = []
                 tokenizer = MOFid_Tokenizer(vocab_file = self.vocab_path)  
                 mofid = open(mofid_path,'r').readlines()
                 for i in range(len(mofid)):
                     token.append(str(tokenizer.tokenize(mofid[i])))
     
                               
              logits_output_path = output_path+'_logits.txt'
              output = open(logits_output_path,'w')
              for i in range(len(logits)):
                    output.write(str(logits[i]))
                    output.write('\n')
              output.close()
     
              if self.Model_Type == 'BertforRegression' and num_labels > 1 :

                 for i in range(num_labels):
                     logit=[]
                     last=len(logits)-num_labels+i+1  
                     for j in np.arange(i,last,num_labels):                         
                         logit.append(logits[j])
                                              
                     logit_output_path = output_path+'_logit_'+str(i+1)+'.txt'
                     np.savetxt(logit_output_path,logit,fmt='%f')
                     
              if output_tsne_data_to_txt == True and output_representation_to_txt == False :  
                 print('Error: output_representation_to_txt == False, the tsne data can not be calculated')
              
              if output_tsne_data_to_txt == True and output_representation_to_txt == True :   
                  
                 print('Warning: the tsne data is being calculated.')
                 
                 tsne_data=[]            
                 tsne_data_output_path = output_path+'_tsne_data_output.txt'            
                 for i in range(len(Representation_output)):
                     tsne_data.append(list(Representation_output[i]))
                 
                 tsne_data_output = TSNE(n_components=2).fit_transform(np.array(tsne_data))
                 
                 print('The tsne data has been calculated.')
                 
                 np.savetxt(tsne_data_output_path,tsne_data_output,fmt='%f') 
                   
                     
              if output_representation_to_txt == True :
                 Representation_output_path = output_path+'_Representation_output.txt'
                 np.savetxt(Representation_output_path,Representation_output)
                 
              if output_attentions_to_txt == True :
                 attention_path_exist = os.path.exists(output_path+'Attettion_Head') 
                 if attention_path_exist == False :
                    os.mkdir(output_path+'Attettion_Head')   
                    
                 for i in range(len(attention[0])):                 
                       attention_output_path = output_path+'Attettion_Head/'+'_attention_Head_'+str(i+1)+'_.txt'
                       np.savetxt(attention_output_path,attention[0][i]) 
     
                 token_output_path = output_path+'Attettion_Head/'+'_token.txt'
                 output = open(token_output_path,'w')
                 for i in range(len(token)):
                     output.write(str(token[i]))
                     output.write('\n')
                 output.close()
                 
              return print('Finish.')