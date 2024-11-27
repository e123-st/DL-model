import torch
from torch import nn
#import torch.nn.functional as F
from torch.nn import MSELoss
from transformers import BertModel
from transformers import BertPreTrainedModel


class BertforRegressionPlusMLP(BertPreTrainedModel):
      
    def __init__(self,
                 config,                 
                 MLP_input_neurons,
                 MLP_output_neurons,
                 MLP_num_hidden_layer,
                 MLP_num_hidden_neurons,
                 MLP_activation,
                 RH_input_neurons,
                 RH_output_neurons,
                 RH_num_hidden_layer,
                 RH_num_hidden_neurons,
                 RH_activation,                 
                 use_Trainer = True,
                 output_representation_to_txt = False,                                               
                 **kwrgs):
        
        
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config) 
        
        self.MLP = MLP(input_neurons = MLP_input_neurons,
                       output_neurons = MLP_output_neurons,
                       num_hidden_layer = MLP_num_hidden_layer,
                       num_hidden_neurons = MLP_num_hidden_neurons,
                       activation = MLP_activation)
                        
        self.RegressionHead = MLP(input_neurons = RH_input_neurons,
                                  output_neurons = RH_output_neurons,
                                  num_hidden_layer = RH_num_hidden_layer,
                                  num_hidden_neurons = RH_num_hidden_neurons,
                                  activation = RH_activation)    
        
        self.use_Trainer = use_Trainer
        
        self.output_representation_to_txt = output_representation_to_txt
        
    def forward(self,
                global_feature = None,
                input_ids = None,
                attention_mask = None,
                token_type_ids = None,
                position_ids = None,
                head_mask = None,
                inputs_embeds = None,
                labels = None,
                output_attentions = None,
                output_hidden_states = None):

        output_tf = self.bert(input_ids = input_ids,
                              attention_mask = attention_mask,
                              token_type_ids = token_type_ids,
                              position_ids = position_ids,
                              head_mask = head_mask,
                              inputs_embeds = inputs_embeds,
                              output_attentions = output_attentions,
                              output_hidden_states = output_hidden_states)
        
        output_mlp = self.MLP(global_feature) 
                                   
        x_cat = torch.cat((output_tf[1],output_mlp),dim=1)    #output_tf[1] : pooled_output
                
        x = self.dropout(x_cat)
        
        logits = self.RegressionHead(x)
        
        if self.use_Trainer == True :
           output = (logits,) + output_tf[2:]  # add hidden states and attention if they are here
        
           loss_fct = MSELoss()        
           loss = loss_fct(logits.view(-1),labels.view(-1))

           output = (loss,) + output   
                   
        if self.use_Trainer == False and self.output_representation_to_txt == False :
           output = (logits,) + output_tf[2:] # add hidden states and attention if they are here
           
        if self.use_Trainer == False and self.output_representation_to_txt == True :
           output = (logits,) + output_tf[2:] # add hidden states and attention if they are here
           output = output + (x_cat,)
        
        return output
        

class BertforRegression(BertPreTrainedModel):
      
    def __init__(self,
                 config,
                 RH_input_neurons,
                 RH_output_neurons,
                 RH_num_hidden_layer,
                 RH_num_hidden_neurons,                 
                 RH_activation,
                 loss_weights = None,
                 use_Trainer = True,
                 output_representation_to_txt = False, 
                 **kwrgs):
        
        
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config) 
        
        self.RegressionHead = MLP(input_neurons = RH_input_neurons,
                                  output_neurons = RH_output_neurons,
                                  num_hidden_layer = RH_num_hidden_layer,
                                  num_hidden_neurons = RH_num_hidden_neurons,
                                  activation = RH_activation)
        
        self.loss_weights = loss_weights
        
        self.use_Trainer = use_Trainer
        
        self.output_representation_to_txt = output_representation_to_txt
        
    def forward(self,
                input_ids = None,
                attention_mask = None,
                token_type_ids = None,
                position_ids = None,
                head_mask = None,
                inputs_embeds = None,
                labels = None,
                output_attentions = None,
                output_hidden_states = None):

        output = self.bert(input_ids = input_ids,
                           attention_mask = attention_mask,
                           token_type_ids = token_type_ids,
                           position_ids = position_ids,
                           head_mask = head_mask,
                           inputs_embeds = inputs_embeds,
                           output_attentions = output_attentions,
                           output_hidden_states = output_hidden_states)

        x_po = output[1]   #pooled_output
        x = self.dropout(x_po)
        logits = self.RegressionHead(x)
        
        if self.use_Trainer == True :
           output = (logits,) + output[2:]  # add hidden states and attention if they are here
        
           loss_fct = MSELoss() 
           
           if self.num_labels == 1:
              loss = loss_fct(logits.view(-1),labels.view(-1))
              
           if self.num_labels > 1:
              if len(self.loss_weights) == self.num_labels :
                 loss = 0
                 for i in range(self.num_labels):
                     loss_split = loss_fct(logits[:,i].view(-1),labels[:,i].view(-1))
                     loss=loss+loss_split*self.loss_weights[i]
              else:
                  print('The sizes of loss_weights are not equal to the num_labels. Please modify.')
                 
           output = (loss,) + output   
       
        if self.use_Trainer == False and self.output_representation_to_txt == False :
           output = (logits,) + output[2:] # add hidden states and attention if they are here
           
        if self.use_Trainer == False and self.output_representation_to_txt == True :
           output = (logits,) + output[2:] # add hidden states and attention if they are here
           output = output + (x_po,)
           
        return output
        


class MLP(nn.Module):
    
      def __init__(self,
                   input_neurons,
                   output_neurons,
                   num_hidden_layer = None,
                   num_hidden_neurons = None,
                   activation = 'relu'):
          
          
          super().__init__()
          self.input_neurons = input_neurons
          self.output_neurons = output_neurons
          self.num_hidden_layer = num_hidden_layer  

          if num_hidden_layer == None and activation != None :
             print('Caution: There are not hidden layer.The setting for activation does not work.')
             print("----------------------------------------------")
                         
          if num_hidden_layer != None and activation == None :
             print('Caution: The activation fuction was not setted in hidden layer.')
             print("----------------------------------------------")
             
          if num_hidden_neurons == None and activation != None :
             print('Caution: There are not hidden neurons.The setting for activation does not work.')
             print("----------------------------------------------")   
             
          if num_hidden_neurons != None and activation == None :
             print('Caution: The activation fuction was not setted in hidden layer.')
             print("----------------------------------------------")      
          
          if activation == 'relu' :
             self.activation = nn.ReLU() 
             
          elif activation == 'tanh' :
             self.activation = nn.Tanh() 
             
          elif activation == 'sigmoid' :
             self.activation = nn.Sigmoid() 
             
          #elif activation == 'leakyrelu' :
          #   self.activation = nn.LeakyReLU()
             
          model = nn.ModuleList([])
          
          if num_hidden_neurons != None and self.num_hidden_layer == None : 
             print("Caution: There are not hidden layer.The setting for num_hidden_neurons does not work.")
             print("----------------------------------------------")
             
          if self.num_hidden_layer == None and num_hidden_neurons == None :
             model.append(nn.Linear(input_neurons,output_neurons,bias=True))   
           
          if num_hidden_neurons != None and self.num_hidden_layer != None :                                                         
             if len(num_hidden_neurons) == self.num_hidden_layer  :
                self.neurons = num_hidden_neurons
                model.append(nn.Linear(input_neurons,self.neurons[0],bias=True)) 
                            
                for i in range(self.num_hidden_layer-1):                 
                    model.append(self.activation)  
                    model.append(nn.Linear(self.neurons[i],self.neurons[i+1],bias=True))  
                
                model.append(self.activation)
                model.append(nn.Linear(self.neurons[self.num_hidden_layer-1],output_neurons,bias=True))
                
             else:   
                print("Error: The dimensions of num_hidden_neurons are not matched with the num_hidden_layer.")
                print("----------------------------------------------")                
        
          self.model=nn.Sequential(*model)   
          
      def forward(self,x):          
          return self.model(x)

        


