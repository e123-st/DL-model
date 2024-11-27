from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
from transformers import BertTokenizer
import torch
import numpy as np


class MOFid_Tokenizer(BertTokenizer):
  
  def __init__(self,vocab_file: str,**kwargs):

      super().__init__(vocab_file, do_lower_case = False, **kwargs)   

      self.smiles_tokenizer = BasicSmilesTokenizer()
      self.topo_tokenizer = TopoTokenizer()

  def tokenize(self, text: str): 
      
      """
      Example:
          
      N1=C[C](C=N1)C=CC1=C[N]N=C1.[N]1N=CC(=C1)C=CC1=C[N]N=C1.[Ni] MOFid-v1.bcu.cat0;ARC_MOF_1
      
      MOFid_Type:
          
          SMILES MOFid-v1.Topology.cat;MOF_name 
          
          ---->  SMILES&&Topology.cat
      """
      
      text = text.strip('\n').strip('"')
      MOFid,MOF_name = text.split(';')
      MOFid = MOFid.replace(' MOFid-v1.','&&')      
      smiles,topo = MOFid.split('&&')
      smiles_tokens = self.smiles_tokenizer.tokenize(smiles)
      topo_tokens = self.topo_tokenizer.tokenize(topo)
      MOFid_tokens = smiles_tokens+['&&']+topo_tokens 
      
      return MOFid_tokens 
     
class TopoTokenizer(object):
  
  def __init__(self):
      
      return

  def tokenize(self, text):
      
      topo_cat = text.split('.')
      if len(topo_cat) < 2:
         topos_or_cat = topo_cat[0]
         topos_or_cat = topos_or_cat.split(',')
         tokens = topos_or_cat
      else:
         topos, cat = topo_cat[0], topo_cat[1]
         topos = topos.split(',')
         tokens = topos + [cat]
         
      return tokens


class token_encode(torch.utils.data.Dataset):
    
    def __init__(self,data_path:str,vocab_path:str,max_length:int): 
        
        self.file = open(data_path,'r').readlines()
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.max_length = max_length
        self.num_sample = len(self.file)
        
    def __len__(self):
        
        return self.num_sample
    
    def __getitem__(self,index):    
           
        tokenizer = MOFid_Tokenizer(vocab_file = self.vocab_path)
        encode = tokenizer.encode_plus(text = str(self.file[index]),
                                       max_length = self.max_length,
                                       padding = 'max_length', 
                                       truncation = True,
                                       return_tensors = 'pt')   
        
        input_data = {'input_ids':encode['input_ids'].flatten(),
                      'token_type_ids':encode['token_type_ids'].flatten(),
                      'attention_mask':encode['attention_mask'].flatten()}     
        
        return input_data

    
class token_encode_with_labels(torch.utils.data.Dataset):
    
    def __init__(self,data_x_path:str,data_y_path:str,vocab_path:str,max_length:int,fp16=False): 
        
        self.file_x = open(data_x_path,'r').readlines()
        self.file_y = open(data_y_path,'r').readlines()
        #self.data_path = data_x_path
        self.vocab_path = vocab_path
        self.max_length = max_length
        self.num_sample = len(self.file_x)
        if fp16 == False :
           self.precision = 'fp32'
        if fp16 == True :
           self.precision = 'fp16'
        
    def __len__(self):
        return self.num_sample
    
    def __getitem__(self,index):               
        tokenizer = MOFid_Tokenizer(vocab_file = self.vocab_path)
        encode = tokenizer.encode_plus(text = str(self.file_x[index]),
                                       max_length = self.max_length,
                                       padding = 'max_length', 
                                       truncation = True,
                                       return_tensors = 'pt')  
        labels = self.file_y[index]
        labels = labels.strip('\n')
        
        if len(labels) == 1 :
           labels = [float(labels)]
           l = torch.tensor(labels)
        
        if len(labels) > 1 :
           labels = labels.split(' ')
           l = []
           for i in range(len(labels)):
               A=float(labels[i])
               l.append(A)          
           l = np.array(l)
           
        if self.precision == 'fp32' :
           labels = torch.from_numpy(l).to(torch.float32) 
        if self.precision == 'fp16' :
           labels = torch.from_numpy(l).to(torch.float16) 
        
        input_data = {'input_ids':encode['input_ids'].flatten(),
                      'token_type_ids':encode['token_type_ids'].flatten(),
                      'attention_mask':encode['attention_mask'].flatten(),
                      'labels':labels} 
        
        return input_data    
    
class token_encode_with_MLPinputs(torch.utils.data.Dataset):
    
    def __init__(self,data_gf_path:str,data_mofid_path:str,vocab_path:str,max_length:int,fp16=False): 
        
        self.file_gf = open(data_gf_path,'r').readlines()
        self.file_mofid = open(data_mofid_path,'r').readlines()        
        #self.data_gf_path = data_gf_path
        #self.data_mofid_path = data_mofid_path
        self.vocab_path = vocab_path
        self.max_length = max_length
        self.num_sample = len(self.file_gf)
        if fp16 == False :
           self.precision = 'fp32'
        if fp16 == True :
           self.precision = 'fp16'   
        
    def __len__(self):
        return self.num_sample
    
    def __getitem__(self,index):        
        
#Global feature        
        global_feature = self.file_gf[index] 
        global_feature = global_feature.strip('\n')
        global_feature = global_feature.split(' ')
        gf = []
        for i in range(len(global_feature)):
              A=float(global_feature[i])
              gf.append(A)          
        global_feature = np.array(gf)
        #global_feature = np.array(global_feature)
        if self.precision == 'fp32' :
           global_feature = torch.from_numpy(global_feature).to(torch.float32) 
        if self.precision == 'fp16' :
           global_feature = torch.from_numpy(global_feature).to(torch.float16) 
        
#MOFid        
        tokenizer = MOFid_Tokenizer(vocab_file = self.vocab_path)
        encode = tokenizer.encode_plus(str(self.file_mofid[index]),
                                       max_length = self.max_length,
                                       padding = 'max_length', 
                                       truncation = True,
                                       return_tensors = 'pt')  
     
        
        input_data = {'global_feature':global_feature,
                      'input_ids':encode['input_ids'].flatten(),
                      'token_type_ids':encode['token_type_ids'].flatten(),
                      'attention_mask':encode['attention_mask'].flatten()}
                      
        
        return input_data

    
class token_encode_with_MLPinputs_and_Regresslabels(torch.utils.data.Dataset):
    
    def __init__(self,data_gf_path:str,data_mofid_path:str,data_y_path:str,vocab_path:str,max_length:int,fp16=False): 
        
        self.file_gf = open(data_gf_path,'r').readlines()
        self.file_mofid = open(data_mofid_path,'r').readlines()
        self.file_y = open(data_y_path,'r').readlines()
        #self.data_gf_path = data_gf_path
        #self.data_mofid_path = data_mofid_path
        #self.data_y_path = data_y_path
        self.vocab_path = vocab_path
        self.max_length = max_length
        self.num_sample = len(self.file_gf)
        if fp16 == False :
           self.precision = 'fp32'
        if fp16 == True :
           self.precision = 'fp16'   
        
    def __len__(self):
        return self.num_sample
    
    def __getitem__(self,index):        
        
#Global feature        
        global_feature = self.file_gf[index] 
        global_feature = global_feature.strip('\n')
        global_feature = global_feature.split(' ')
        gf = []
        for i in range(len(global_feature)):
              A=float(global_feature[i])
              gf.append(A)          
        global_feature = np.array(gf)
        #global_feature = np.array(global_feature)
        if self.precision == 'fp32' :
           global_feature = torch.from_numpy(global_feature).to(torch.float32) 
        if self.precision == 'fp16' :
           global_feature = torch.from_numpy(global_feature).to(torch.float16) 
        
#MOFid        
        tokenizer = MOFid_Tokenizer(vocab_file = self.vocab_path)
        encode = tokenizer.encode_plus(str(self.file_mofid[index]),
                                       max_length = self.max_length,
                                       padding = 'max_length', 
                                       truncation = True,
                                       return_tensors = 'pt')  
#labels        
        labels = self.file_y[index] 
        labels = labels.strip('\n')
        labels = [float(labels)]
        labels = torch.tensor(labels)
        
        input_data = {'global_feature':global_feature,
                      'input_ids':encode['input_ids'].flatten(),
                      'token_type_ids':encode['token_type_ids'].flatten(),
                      'attention_mask':encode['attention_mask'].flatten(),
                      'labels':labels} 
        
        return input_data    

    