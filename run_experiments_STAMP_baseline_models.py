# coding=utf-8
from optparse import OptionParser
import tensorflow as tf
import pandas as pd
import numpy as np
from STAMP.data_prepare.entity.samplepack import Samplepack
from STAMP.data_prepare.load_dict import load_random
from STAMP.data_prepare.cikm16data_read import load_data2
from STAMP.data_prepare.rsyc15data_read_p import load_data_p
from STAMP.util.Config import read_conf
from STAMP.util.FileDumpLoad import dump_file, load_file
from STAMP.util.Randomer import Randomer
import json
from pathlib import Path
# the data path.
import tensorflow as p
project_name = '/STAMP'

path_cikm = 'data/cikm16/train-item-views.csv'
mid_cikm16_emb_dict = "cikm16_emb_dict.data"

data_path = Path("data/")
data_path = data_path.resolve()
result_path = Path("results/")
result_path = result_path.resolve()

rsc15_name = "yoochoose-clicks.dat"
digi_name = "train-item-views.csv"

def load_tt_datas(config={}):
    if config['dataset'] == 'rsc15_4':
            print(config['dataset'])
            
            rsc15_ratio = 4
            dataset_name = "yoochoose-clicks.dat"
            result_path = Path("data/session_based_datasets/")
            result_path = result_path.resolve()

            train_data, test_data, item2idx, n_items = load_data_p(data_path/rsc15_name, rsc15_ratio)
            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            
        
    if config['dataset'] == 'rsc15_64':
            print(config['dataset'])
            rsc15_ratio = 64
            dataset_name = "yoochoose-clicks.dat"
            result_path = Path("data/session_based_datasets/")
            result_path = result_path.resolve()
            train_data, test_data, item2idx, n_items = load_data_p(data_path/rsc15_name,rsc15_ratio)
            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx, edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            
    if config['dataset'] == 'digi':
            train_data, test_data, item2idx, n_items = load_data2(data_path/digi_name, class_num=config['class_num'])
            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            path = 'datas/mid_data'        
    return train_data, test_data

def load_conf(model, modelconf):
    name="model.conf"
    model_conf = read_conf(model, modelconf/ name)
    if model_conf is None:
        raise Exception("wrong model config path.", model_conf)
    module = model_conf['module']
    obj = model_conf['object']
    params = model_conf['params']
    params = params.split("/")
    paramconf = ""
    model = params[-1]
    for line in params[:-1]:
        paramconf += line + "/"
    paramconf = paramconf[:-1]
    # load super params.
    name = "nn_param.conf"
    param_conf = read_conf(model, modelconf / name)
    return module, obj, param_conf
def option_parse(dataset = "digi"):
    global rsc15_ratio
    '''
    parse the option.
    '''
    parser = OptionParser()
    if dataset == "digi":
        parser.add_option("-m","--model", action='store',type='string',dest="model", default='stamp_cikm')
        parser.add_option("-d","--dataset", action='store', type='string', dest="dataset", default='digi')
        
    if dataset =="rsc15_64":
        parser.add_option("-m","--model", action='store',type='string',dest="model", default='stamp_rsc')
        parser.add_option("-d","--dataset", action='store', type='string', dest="dataset", default='rsc15_64')
        rsc15_ratio = 64

    if dataset =="rsc15_4":
        parser.add_option("-m","--model", action='store',type='string',dest="model", default='stamp_rsc')
        parser.add_option("-d","--dataset", action='store', type='string', dest="dataset", default='rsc15_4')

    parser.add_option("-r","--reload",action='store_true',dest="reload",default=True)
    parser.add_option("-c","--classnum",action='store',type='int',dest="classnum",default=3)

    parser.add_option("-a","--nottrain",action='store_true',dest="not_train",default=False)
    parser.add_option("-n","--notsavemodel",action='store_true',dest="not_save_model",default=False)
    parser.add_option("-p","--modelpath",action='store',type='string',dest="model_path",default='/home/herb/code/WWW18/ckpt/seq2seqlm.ckpt-3481-201709251759-lap')
    parser.add_option("-i","--inputdata",action='store',type='string',dest="input_data",default='test')
    parser.add_option("-e","--epoch",action='store',type='int',dest="epoch",default=30)
    (option, args) = parser.parse_args()
    return option
def run_experiments_for_STAMP(options, ):
    print("<<<<<<<<<<<<<<<<<<<< Wait for results >>>>>>>>>>>>>>>>>>>>>>>")
    model = options.model
    dataset = options.dataset
    class_num = options.classnum
    is_train = not options.not_train
    is_save = not options.not_save_model
    model_path = options.model_path
    input_data = options.input_data
    epoch = 1 #options.epoch
    print("Number of Epochs: "+str(epoch))
    path = Path("STAMP/config/")
    path = path.resolve()
    module, obj, config = load_conf(model, path)
    config['model'] = model
    config['dataset'] = dataset
    config['class_num'] = class_num
    config['nepoch'] = epoch
    train_data, test_data = load_tt_datas(config)
    path = "STAMP."+module.split(".")[0]+"."+module.split(".")[1]
    module = __import__(path, fromlist=True)
    # setup randomer
    Randomer.set_stddev(config['stddev'])
    with tf.Graph().as_default():
        # build model
        model = getattr(module, obj)(config)
        model.build_model()
        if is_save or not is_train:
            saver = tf.train.Saver(max_to_keep=30)
        else:
            saver = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(dataset)
            if dataset == "digi":
                mmr, recall = model.train(sess, train_data, test_data, saver)

                df = pd.DataFrame()
                df["MRR@20"] = [mmr]
                df["Rec@20"] = [recall]
                
                name = "STAMP_"+config['dataset']+".txt"

                df.to_csv(data_path/name, index = False, sep = "\t")
            else:
                mmr, recall = model.train(sess, train_data, test_data, saver)
                df = pd.DataFrame()
                df["MRR@20"] = [mmr]
                df["Rec@20"] = [recall]
                
                name = "STAMP_"+config['dataset']+".txt"
                df.to_csv(data_path/name, index = False, sep = "\t")  

# import baseline models.....
from STAMP.baselines.stan.main_stan import *
from STAMP.baselines.vstan.main_vstan import *
from STAMP.baselines.sfcknn.main_sfcknn import *
from STAMP.baselines.SR.main_sr import *

if __name__ == '__main__':

    dataset =     "digi" #"digi", rsc15_4, rsc15_64
    options = option_parse(dataset = dataset)
    print("<<<<<<<<<<<<<<<<<<<<<<  DATASET:  "+options.dataset+">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    topKList = [10, 20]
    print("Experiments are runinig for SR model................... wait for results...............")
    se_obj = SequentialRulesMain(data_path, result_path, dataset = options.dataset)
    se_obj.fit_(topKList)
    print("Experiments are runinig for SFCKNN model................... wait for results...............")
    sfcknn_obj = SFCKNN_MAIN(data_path, result_path, dataset = options.dataset)
    sfcknn_obj.fit_(topKList)
    print("Experiments are runinig for STAN model................... wait for results...............")
    stan_obj = STAN_MAIN(data_path, result_path, dataset = options.dataset)
    stan_obj.fit_(topKList)
    
    print("Experiments are runinig for VSTAN model................... wait for results...............")
    vstan_obj = VSTAN_MAIN(data_path, result_path, dataset = options.dataset)
    vstan_obj.fit_(topKList)
    print("Run experiments for STAMP model")
    run_experiments_for_STAMP(options)
    



