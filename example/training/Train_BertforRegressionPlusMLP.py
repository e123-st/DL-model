from transformer.Transformer_Encoder import Models

basic_path=r'XXX'

BertPretrainModel_path=basic_path+'/model/[-L_3-A_12-]_Pretrain_model'

vocab_path=basic_path+'/tokenizer/vocab_full.txt'

train_gf_path=basic_path+'/data/train_gf.txt'

eval_gf_path=basic_path+'/data/eval_gf.txt'

train_mofid_path=basic_path+'/data/train_mofid.txt'

eval_mofid_path=basic_path+'/data/eval_mofid.txt'

train_labels_path=basic_path+'/data/train_y.txt'

eval_labels_path=basic_path+'/data/eval_y.txt'

output_path=basic_path+'/model/'

model_save_path=basic_path+'/model/[-L_3-A_12-]_mofid_gf_isotherm_model'

model=Models(Model_Type = 'BertforRegressionPlusMLP',
             vocab_path = vocab_path,    
             output_dir = output_path,                                
             max_len = 512,
             use_GPU = True)

model.train(MLP_input_neurons = 8,
            MLP_output_neurons = 48,
            MLP_num_hidden_layer = 3,
            MLP_num_hidden_neurons = [48,48,48],
            MLP_activation = 'relu',
            Head_input_neurons = 816,
            Head_output_neurons = 1,
            Head_num_hidden_layer = 3,
            Head_num_hidden_neurons = [1224,1224,1224],
            Head_activation = 'relu',
            train_batch_size = 32, 
            eval_batch_size = 32,
            learning_rate = 1e-4,
            num_train_epochs = 100,
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
            train_gf_path = train_gf_path,
            eval_gf_path = eval_gf_path,
            train_labels_path = train_labels_path,
            eval_labels_path = eval_labels_path,                              
            num_labels = 1,
            KeepTraining = False)