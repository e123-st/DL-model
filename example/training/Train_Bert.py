from transformer.Transformer_Encoder import Models

num_hidden_layers = 3
num_attention_heads = 12

basic_path=r'xxx'

vocab_path=basic_path+'/tokenizer/vocab_full.txt'

train_mofid_path=basic_path+'/data/train_mofid.txt'

eval_mofid_path=basic_path+'/data/eval_mofid.txt'

output_path=basic_path+'/model'

model_save_path=basic_path+'/model/'+'[-L_'+str(num_hidden_layers)+'-A_'+str(num_attention_heads)+'-]'+'_Pretrain_model'

model=Models(Model_Type = 'Bert',
             vocab_path = vocab_path,    
             output_dir = output_path,                                
             max_len = 512,
             use_GPU = True)

model.train(num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,            
            train_batch_size = 32, 
            eval_batch_size = 32,
            learning_rate = 1e-4,
            num_train_epochs = 20,
            weight_decay = 0,
            warmup_ratio = 0,
            evaluation_strategy = 'epoch',
            save_strategy = 'epoch',
            save_steps = 500,
            eval_steps = 500,
            save_safetensors = False,
            load_best_model_at_end = False,
            metric_for_best_model = 'eval_loss',
            greater_is_better = False,
            model_save_path = model_save_path,                        
            train_mofid_path = train_mofid_path,
            eval_mofid_path = eval_mofid_path,
            vocab_size = 5141, 
            hidden_size = 768,
            intermediate_size = 3072, 
            hidden_act = "gelu", 
            hidden_dropout_prob = 0.1, 
            attention_probs_dropout_prob = 0.1,
            type_vocab_size = 1, 
            initializer_range = 0.02, 
            layer_norm_eps = 1e-12, 
            pad_token_id = 0, 
            bos_token_id = 2, 
            eos_token_id = 3, 
            position_embedding_type = "absolute", 
            use_cache = True, 
            classifier_dropout = None)