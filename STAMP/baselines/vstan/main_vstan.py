# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:18:04 2024
@author: shefai
"""
from STAMP.data_prepare.rsyc15data_read_p import *
from STAMP.data_prepare.cikm16data_read import *
from STAMP.baselines.vstan.vstan  import *
from pathlib import Path
from STAMP.accuracy_measures import *
from tqdm import tqdm

class VSTAN_MAIN:
    def __init__(self, data_path, result_path, dataset = "digi"):
        self.dataset = dataset
        self.result_path = result_path
        if dataset == "digi":
            self.k = 2000
            self.sample_size = 5500
            self.lambda_spw = 0.104
            self.lambda_snh = 53
            self.lambda_inh = 2.4
            self.lambda_idf = 1
            
            name = "train-item-views.csv"
            self.train_data, self.test_data = readAndSplitData_cikm(data_path / name)

            self.train_data.rename(columns = {'sessionId':'SessionId', 'itemId': 'ItemId'}, inplace = True)
            self.test_data.rename(columns = {'sessionId':'SessionId', 'itemId': 'ItemId'}, inplace = True)

            self.unique_items_ids = self.train_data["ItemId"].unique()
            
            
        elif dataset == 'rsc15_4':
            # vstan-k=5500-sample_size=3500-lambda_spw=0.106-lambda_snh=56-lambda_inh=3.8
            self.k = 5500
            self.sample_size = 3500
            self.lambda_spw = 0.106
            self.lambda_snh = 56
            self.lambda_inh = 3.8
            self.lambda_idf = 1
            
            
            name = "yoochoose-clicks.dat"
            self.train_data, self.test_data = readAndSplitData_rsc15(data_path / name, 4)

            self.unique_items_ids = self.train_data["ItemId"].unique()
            
            
            
        elif dataset == "rsc15_64":
            
            # vstan-k=500-sample_size=2500-lambda_spw=0.94-lambda_snh=55-lambda_inh=1.5
            self.k = 500
            self.sample_size = 2500
            self.lambda_spw = 0.94
            self.lambda_snh = 55
            self.lambda_inh = 1.5
            self.lambda_idf = 1
            
            name = "yoochoose-clicks.dat"
            self.train_data, self.test_data = readAndSplitData_rsc15(data_path / name, 64)
            self.unique_items_ids = self.train_data["ItemId"].unique()
            
            
            
        else:
            print("Mention your datatypes")
            
            
    def fit_(self, topKList):
        
        obj1 = VSKNN_STAN(k = self.k,  sample_size = self.sample_size, lambda_spw = self.lambda_spw, lambda_snh = self.lambda_snh, lambda_inh = self.lambda_inh, lambda_idf = self.lambda_idf )
        obj1.fit(self.train_data)
        
        session_key ='SessionId'
        time_key='Time'
        item_key= 'ItemId'
        
        # Intialize accuracy measures.....
        MRR_dictionary = dict()
        for i in topKList:
            MRR_dictionary["MRR_"+str(i)] = MRR(i)
            
        Recall_dictionary = dict()
        for i in topKList:
            Recall_dictionary["Recall_"+str(i)] = Recall(i)
        
        test_data = self.test_data
        test_data.sort_values([session_key, time_key], inplace=True)
        items_to_predict = self.unique_items_ids
        # Previous item id and session id....
        prev_iid, prev_sid = -1, -1
        
        print("Starting predicting")
        for i in tqdm(range(len(test_data))):
            sid = test_data[session_key].values[i]
            iid = test_data[item_key].values[i]
            ts = test_data[time_key].values[i]
            
            if prev_sid != sid:
                # this will be called when there is a change of session....
                prev_sid = sid
            else:
                # prediction starts from here.......
                preds = obj1.predict_next(sid, prev_iid, items_to_predict, ts)
                preds[np.isnan(preds)] = 0
    #             preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
                preds.sort_values( ascending=False, inplace=True )    
    
                for key in MRR_dictionary:
                    MRR_dictionary[key].add(preds, iid)
                    
                
                # Calculate the Recall values
                for key in Recall_dictionary:
                    Recall_dictionary[key].add(preds, [iid])
                
            prev_iid = iid
            
        # get the results of MRR values.....
        result_frame = pd.DataFrame()    
        for key in MRR_dictionary:
            print(key +"   "+ str(  MRR_dictionary[key].score()    ))
            result_frame[key] =   [MRR_dictionary[key].score()]
            
            
        # get the results of MRR values.....    
        for key in Recall_dictionary:
            print(key +"   "+ str(  Recall_dictionary[key].score()    ))
            result_frame[key] = [Recall_dictionary[key].score()]
        
        
        name = "STAMP_VSTAN_"+self.dataset+".txt"
        result_frame.to_csv(self.result_path /  name, sep = "\t", index = False) 
        
       
        
        
        
        
        


