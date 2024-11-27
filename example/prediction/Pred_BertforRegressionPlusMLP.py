from transformer.Transformer_Encoder import Models

basic_path=r'XXX'

model_path=basic_path+'/model/[-L_3-A_12-]_mofid_gf_isotherm_model'

vocab_path=basic_path+'/tokenizer/vocab_full.txt'

gf_path=basic_path+'/gf.txt'

mofid_path=basic_path+'/mofid.txt'

label_path=basic_path+'/label.txt'

output_path=basic_path+'/output_file/'


model=Models(Model_Type = 'BertforRegressionPlusMLP',
             vocab_path = vocab_path, 
             #output_dir = output_path,                                             
             max_len = 512,
             use_GPU = True,  
             use_Trainer = False,
             use_Accelerate = False)


model.predict(model = model_path,
           batch_size = 32,
           MLP_input_neurons = 8,
           MLP_output_neurons = 48,
           MLP_num_hidden_layer = 3,
           MLP_num_hidden_neurons = [48,48,48],
           MLP_activation = 'relu',
           Head_input_neurons = 816,
           Head_output_neurons = 1,
           Head_num_hidden_layer = 3,
           Head_num_hidden_neurons = [1224,1224,1224],
           Head_activation = 'relu',
           gf_path = gf_path,
           mofid_path = mofid_path,
           label_path = label_path,
           output_path = output_path,                          
           split = 'Prefix',
           output_hidden_states = True,
           output_attentions = True, 
           output_representation_to_txt= False, 
           output_attentions_to_txt = False,
           num_labels = 1,
           num_loader_workers = 0)