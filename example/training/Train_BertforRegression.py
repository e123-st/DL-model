from transformer.Transformer_Encoder import Models

basic_path=r'xxx'

BertPretrainModel_path=basic_path+'/model/[-L_3-A_12-]_Pretrain_model'

vocab_path=basic_path+'/tokenizer/vocab_full.txt'

train_mofid_path=basic_path+'/data/train_mofid.txt'

eval_mofid_path=basic_path+'/data/eval_mofid.txt'

train_labels_path=basic_path+'/data/train_y.txt'

eval_labels_path=basic_path+'/data/eval_y.txt'

output_path=basic_path+'/model/'

model_save_path=basic_path+'/model/[-L_3-A_12-]_mofid_ASA_VF_PV_model'

model=Models(Model_Type = 'BertforRegression',
             vocab_path = vocab_path,    
             output_dir = output_path,                                
             max_len = 512,
             use_GPU = True)

model.train(Head_input_neurons = 768,
            Head_output_neurons = 3,
            Head_num_hidden_layer = 3,
            Head_num_hidden_neurons = [1152,1152,1152],
            Head_activation = 'relu',
            loss_weights=[1,1,1],
            train_batch_size = 32, 
            eval_batch_size = 32,
            learning_rate = 1e-4,
            num_train_epochs = 200,
            weight_decay = 0,
            warmup_ratio =  0,
            evaluation_strategy = 'epoch',
            save_strategy = 'epoch',
            save_steps = 500,
            eval_steps = 500,
            save_safetensors = False,
            load_best_model_at_end = False,
            metric_for_best_model = 'eval_loss',
            greater_is_better = False,
            BertPretrainModel_path = BertPretrainModel_path,
            model_save_path = model_save_path,                        
            train_mofid_path = train_mofid_path,
            eval_mofid_path = eval_mofid_path,
            train_labels_path = train_labels_path,
            eval_labels_path = eval_labels_path,                              
            num_labels = 3,
            KeepTraining = False)