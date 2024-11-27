from transformer.Transformer_Encoder import Models

basic_path=r'XXX'

model_path=basic_path+'/model/[-L_3-A_12-]_mofid_ASA_VF_PV_model'

vocab_path=basic_path+'/tokenizer/vocab_full.txt'

mofid_path=basic_path+'/mofid.txt'

label_path=basic_path+'/label.txt'

output_path=basic_path+'/output_file/'


model=Models(Model_Type = 'BertforRegression',
             vocab_path = vocab_path,    
            #output_dir = output_path,                                
             max_len = 512,
             use_GPU = True,  
             use_Trainer = False,
             use_Accelerate = False)


model.eval(model = model_path,
           batch_size = 32,
           Head_input_neurons = 768,
           Head_output_neurons = 3,
           Head_num_hidden_layer = 3,
           Head_num_hidden_neurons = [1152,1152,1152],
           Head_activation = 'relu',
           mofid_path = mofid_path,
           label_path = label_path,
           output_path = output_path,                          
           split = 'Prefix',
           output_hidden_states = True,
           output_attentions = True, 
           output_representation_to_txt= False, 
           output_attentions_to_txt = False,
           output_tsne_data_to_txt = False,
           num_labels = 3,
           num_loader_workers = 0)