import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Train_Eval_Test_split():
      def __init__(self,
                   basic_path,
                   filename,
                   data_type,
                   sheet_name= 0,
                   train_size  = 0.8,
                   eval_size =0.1,
                   test_size = 0.1,
                   random_state = 0,
                   num_data = 10,
                   label = '70'):
          
          """
            basic_path: the directory of output and input file, end with '/', the file name should not be written. str
            
            filname : the name of input file, str
            
            data_type : 'mofid_eval' : for pre-train process without test set
                        'mofid' ： for pre-train process , if the test set in need
                        'mofid_with_label','mofid_with_multiregresslabels': for data with mofid and label 
                        'mofid_gf_with_Regresslabel' : for data with mofid, continous data, and labels
                        'Split_Only_mofid_gf_with_label', 'Split_Only_mofid_with_labels' : just split the data from excel, but not split into training, validtion, and test set
                        
            sheet_name : the serial number in Excel, only for data_type = 'mofid_with_label','mofid_with_multiregresslabels','mofid_gf_with_label', default to 0
                           
            train_size ：the proportion of training set, default to 0.8
                        
            eval_size : the proportion of validation set, default to 0.1
                        
            test_size : the proportion of test set, default to 0.1
                                
            random_state : the random number for data split, default to 0
            
            num_data : the number of column，only if data_type = 'mofid_with_multiregresslabels','mofid_gf_with_label', default to 10
            
            label : the prefix name of output file, only if data_type = 'Split_Only_mofid_gf_with_label' ,'Split_Only_mofid_with_labels'  
            
            Specially:
            1. if data_type = 'mofid_eval', the eval_size and test_size are not need to be set.
            2. if data_type = 'mofid_eval' or 'mofid'的, the input file should be a txt file.
          """
            
          self.dataset_path = basic_path + filename
          self.sheet_name = sheet_name
          self.train_size = train_size
          self.test_size = test_size/(test_size + eval_size)
          self.random_state = random_state
          self.num_data = num_data
          self.label = label
          
          if data_type == 'mofid_eval' :
             
             #训练集&测试集&验证集数据路径
             train_data_path=basic_path+'/train_mofid.txt'
             eval_data_path=basic_path+'/eval_mofid.txt'

             #分割训练集，测试集，验证集
             dataset=open(self.dataset_path,'r').readlines()
             train_data,eval_data=train_test_split(dataset,train_size=self.train_size,random_state=self.random_state)             

             output_train_data=open(train_data_path,'w')
             output_eval_data=open(eval_data_path,'w')

             for i in range(len(train_data)):
                 output_train_data.write(str(train_data[i])) 
             output_train_data.close()

             for j in range(len(eval_data)):
                 output_eval_data.write(str(eval_data[j]))  
             output_eval_data.close()    

             
          if data_type == 'mofid' :
             
             #训练集&测试集&验证集数据路径
             train_data_path=basic_path+'/train_mofid.txt'
             eval_data_path=basic_path+'/eval_mofid.txt'
             test_data_path=basic_path+'/test_mofid.txt'

             #分割训练集，测试集，验证集
             dataset=open(self.dataset_path,'r').readlines()            
             train_data,evaldation_test_data=train_test_split(dataset,train_size=self.train_size,random_state=self.random_state)
             eval_data,test_data=train_test_split(evaldation_test_data,test_size=self.test_size,random_state=self.random_state)

             output_train_data=open(train_data_path,'w')
             output_eval_data=open(eval_data_path,'w')
             output_test_data=open(test_data_path,'w')

             for i in range(len(train_data)):
                 output_train_data.write(str(train_data[i])) 
             output_train_data.close()

             for j in range(len(eval_data)):
                 output_eval_data.write(str(eval_data[j]))  
             output_eval_data.close()    

             for k in range(len(test_data)):
                 output_test_data.write(str(test_data[k])) 
             output_test_data.close()   
          
          if data_type == 'mofid_with_label' :
              
             #训练集&测试集&验证集数据路径
             train_data_path=basic_path+'/train_mofid.txt'
             eval_data_path=basic_path+'/eval_mofid.txt'
             test_data_path=basic_path+'/test_mofid.txt'

             train_y_path=basic_path+'/train_y.txt'
             eval_y_path=basic_path+'/eval_y.txt'
             test_y_path=basic_path+'/test_y.txt'

             #分割训练集，测试集，验证集
             dataset=pd.read_excel(io=self.dataset_path,sheet_name = self.sheet_name)
             x=dataset.iloc[:,1]
             y=dataset.iloc[:,2]
             train_data,valdation_test_data,train_y,valdation_test_y=train_test_split(x,y,train_size=self.train_size,random_state=self.random_state)
             eval_data,test_data,eval_y,test_y=train_test_split(valdation_test_data,valdation_test_y,test_size=self.test_size,random_state=self.random_state)

             #转换格式
             train_data=np.array(train_data)
             eval_data=np.array(eval_data)
             test_data=np.array(test_data)
             
             train_y=np.array(train_y)
             eval_y=np.array(eval_y)
             test_y=np.array(test_y)

             #保存数据
             output_train_data=open(train_data_path,'w')
             output_eval_data=open(eval_data_path,'w')
             output_test_data=open(test_data_path,'w')

             for i in range(len(train_data)):
                 output_train_data.write(str(train_data[i])) 
                 output_train_data.write('\n')
             output_train_data.close()

             for j in range(len(eval_data)):
                 output_eval_data.write(str(eval_data[j]))  
                 output_eval_data.write('\n')
             output_eval_data.close()    

             for k in range(len(test_data)):
                 output_test_data.write(str(test_data[k])) 
                 output_test_data.write('\n') 
             output_test_data.close()

             np.savetxt(train_y_path,train_y,fmt='%f')
             np.savetxt(eval_y_path,eval_y,fmt='%f')
             np.savetxt(test_y_path,test_y,fmt='%f')
          
          if data_type == 'mofid_with_multiregresslabels' :
              
             #训练集&测试集&验证集数据路径
             train_data_path=basic_path+'/train_mofid.txt'
             eval_data_path=basic_path+'/eval_mofid.txt'
             test_data_path=basic_path+'/test_mofid.txt'

             train_y_path=basic_path+'/train_y.txt'
             eval_y_path=basic_path+'/eval_y.txt'
             test_y_path=basic_path+'/test_y.txt'

             #分割训练集，测试集，验证集
             dataset=pd.read_excel(io=self.dataset_path,sheet_name = self.sheet_name)
             x=dataset.iloc[:,1]
             y=dataset.iloc[:,2:self.num_data+1]
             train_data,valdation_test_data,train_y,valdation_test_y=train_test_split(x,y,train_size=self.train_size,random_state=self.random_state)
             eval_data,test_data,eval_y,test_y=train_test_split(valdation_test_data,valdation_test_y,test_size=self.test_size,random_state=self.random_state)

             #转换格式
             train_data=np.array(train_data)
             eval_data=np.array(eval_data)
             test_data=np.array(test_data)
             
             train_y=np.array(train_y)
             eval_y=np.array(eval_y)
             test_y=np.array(test_y)

             #保存数据
             output_train_data=open(train_data_path,'w')
             output_eval_data=open(eval_data_path,'w')
             output_test_data=open(test_data_path,'w')

             for i in range(len(train_data)):
                 output_train_data.write(str(train_data[i])) 
                 output_train_data.write('\n')
             output_train_data.close()

             for j in range(len(eval_data)):
                 output_eval_data.write(str(eval_data[j]))  
                 output_eval_data.write('\n')
             output_eval_data.close()    

             for k in range(len(test_data)):
                 output_test_data.write(str(test_data[k])) 
                 output_test_data.write('\n') 
             output_test_data.close()

             np.savetxt(train_y_path,train_y,fmt='%f')
             np.savetxt(eval_y_path,eval_y,fmt='%f')
             np.savetxt(test_y_path,test_y,fmt='%f')
             
          if data_type == 'mofid_gf_with_Regresslabel' :    
             
             #训练集&测试集&验证集数据路径
             train_gf_path=basic_path+'/train_gf.txt'
             eval_gf_path=basic_path+'/eval_gf.txt'
             test_gf_path=basic_path+'/test_gf.txt'

             train_mofid_path=basic_path+'/train_mofid.txt'
             eval_mofid_path=basic_path+'/eval_mofid.txt'
             test_mofid_path=basic_path+'/test_mofid.txt'

             train_y_path=basic_path+'/train_y.txt'
             eval_y_path=basic_path+'/eval_y.txt'
             test_y_path=basic_path+'/test_y.txt'

             #分割训练集，测试集，验证集
             dataset=pd.read_excel(io=self.dataset_path,sheet_name = self.sheet_name)
             x=dataset.iloc[:,1:self.num_data]
             y=dataset.iloc[:,self.num_data]
             train_x,valdation_test_x,train_y,valdation_test_y=train_test_split(x,y,train_size=self.train_size,random_state=self.random_state)
             eval_x,test_x,eval_y,test_y=train_test_split(valdation_test_x,valdation_test_y,test_size=self.test_size,random_state=self.random_state)

             #分割数据
             train_gf=train_x.iloc[:,0:self.num_data-2]
             eval_gf=eval_x.iloc[:,0:self.num_data-2]
             test_gf=test_x.iloc[:,0:self.num_data-2]

             train_mofid=train_x.iloc[:,self.num_data-2]
             eval_mofid=eval_x.iloc[:,self.num_data-2]
             test_mofid=test_x.iloc[:,self.num_data-2]

             #转换格式
             train_gf=np.array(train_gf)
             eval_gf=np.array(eval_gf)
             test_gf=np.array(test_gf)

             train_mofid=np.array(train_mofid)
             eval_mofid=np.array(eval_mofid)
             test_mofid=np.array(test_mofid)

             train_y=np.array(train_y)
             eval_y=np.array(eval_y)
             test_y=np.array(test_y)

             #保存数据
             np.savetxt(train_gf_path,train_gf,fmt='%f')
             np.savetxt(eval_gf_path,eval_gf,fmt='%f')
             np.savetxt(test_gf_path,test_gf,fmt='%f')

             np.savetxt(train_y_path,train_y,fmt='%f')
             np.savetxt(eval_y_path,eval_y,fmt='%f')
             np.savetxt(test_y_path,test_y,fmt='%f')

             output_train_mofid=open(train_mofid_path,'w')
             output_eval_mofid=open(eval_mofid_path,'w')
             output_test_mofid=open(test_mofid_path,'w')

             for i in range(len(train_mofid)):
                 output_train_mofid.write(str(train_mofid[i]))
                 output_train_mofid.write('\n')
             output_train_mofid.close()

             for j in range(len(eval_mofid)):
                 output_eval_mofid.write(str(eval_mofid[j]))
                 output_eval_mofid.write('\n')
             output_eval_mofid.close()    

             for k in range(len(test_mofid)):
                 output_test_mofid.write(str(test_mofid[k]))
                 output_test_mofid.write('\n')
             output_test_mofid.close()        
              
             
          if data_type == 'Split_Only_mofid_gf_with_label' :
            
             gf_path=basic_path+'/'+str(self.label)+'_gf.txt'

             mofid_path=basic_path+'/'+str(self.label)+'_mofid.txt'

             y_path=basic_path+'/'+str(self.label)+'_y.txt'
            
             path=basic_path+'/'+filename

             dataset = pd.read_excel(io=path,sheet_name=sheet_name)
            
             x=dataset.iloc[:,1:num_data]
             y=dataset.iloc[:,num_data]

             gf=x.iloc[:,0:num_data-2]
             mofid=x.iloc[:,num_data-2]

             gf=np.array(gf)
             mofid=np.array(mofid)
             y=np.array(y)

             np.savetxt(gf_path,gf,fmt='%f')
             np.savetxt(y_path,y,fmt='%f')

             output_mofid=open(mofid_path,'w')
             for i in range(len(mofid)):
                 output_mofid.write(str(mofid[i]))
                 output_mofid.write('\n')
             output_mofid.close()
             
          if data_type == 'Split_Only_mofid_with_labels' :
                         
             mofid_path=basic_path+'/'+str(self.label)+'_mofid.txt'

             y_path=basic_path+'/'+str(self.label)+'_y.txt'
            
             path=basic_path+'/'+filename

             dataset = pd.read_excel(io=path,sheet_name=sheet_name)
            
             mofid=dataset.iloc[:,1]
             y=dataset.iloc[:,2:num_data+1]

             mofid=np.array(mofid)
             y=np.array(y)

             np.savetxt(y_path,y,fmt='%f')

             output_mofid=open(mofid_path,'w')
             for i in range(len(mofid)):
                 output_mofid.write(str(mofid[i]))
                 output_mofid.write('\n')
             output_mofid.close()   
    
          return print('The datas have been splited.')    