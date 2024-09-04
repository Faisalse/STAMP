import pandas as pd
import numpy as np
from STAMP.data_prepare.entity.sample import Sample
from STAMP.data_prepare.entity.samplepack import Samplepack


from datetime import datetime


def load_data2(path, pad_idx=0, class_num = 3):
    '''
    ret = [contexts, aspects, labels, positions] ,
    context.shape = [len(samples), None], None should be the len(context); 
    aspects.shape = [len(samples), None], None should be the len(aspect);
    labels.shape = [len(samples)]
    positions.shape = [len(samples), 2], the 2 means from and to.
    '''
    
    # the global param.
    items2idx = {}  # the ret
    items2idx['<pad>'] = pad_idx
    idx_cnt = 0
    # load train and test data.....
    train_data, test_data = readAndSplitData_cikm(path)
    train_data, idx_cnt = _load_data(train_data, items2idx, idx_cnt, pad_idx, class_num)
    test_data, idx_cnt = _load_data(test_data, items2idx, idx_cnt, pad_idx, class_num)
    item_num = len(items2idx.keys())
    return train_data, test_data, items2idx, item_num

def readAndSplitData_cikm(path):
    data = pd.read_csv(path, sep =";")
    del data["userId"]
    del data['timeframe']
    data['Time'] = data['eventdate'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    del data['eventdate']

    session_lengths = data.groupby('sessionId').size()
    data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]
    item_supports = data.groupby('itemId').size()
    data = data[np.in1d(data.itemId, item_supports[item_supports>=5].index)]
    session_lengths = data.groupby('sessionId').size()
    data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]
    data.rename(columns = {"sessionId":"SessionId", 'itemId':'ItemId'}, inplace = True) 

    print("...Data info after filtering...")
    print("Number of clicks:   "+str(len(data)))
    print("Number of sessions:   "+  str(  len(data["SessionId"].unique()) ))
    print("Number of items:   "+  str(  len(data["ItemId"].unique()) ))

    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400*7].index
    session_test = session_max_times[session_max_times > tmax-86400*7].index

    train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]

    train.sort_values(by=['SessionId', 'Time'],  inplace = True)
    test.sort_values(by=['SessionId', 'Time'],  inplace = True)

    # to keep evaluation same for all model...... 
    test_seq_f = list()
    test_seq = test.groupby('SessionId')['ItemId'].apply(list).to_dict()
    for key, seq_ in test_seq.items():
        for i_ in range(1, len(seq_)):
            test_seq_f.append(seq_[:-i_] +  [seq_[-i_]])
    # build a dataset.....
    new_session_id = list()
    item_list = list()
    count = 1
    for list_ in test_seq_f:
        for item_ in list_:
            item_list.append(item_)
            new_session_id.append(count)
        count += 1
    df = pd.DataFrame()
    df["SessionId"]   = new_session_id
    df["ItemId"]   = item_list
    return train, df

def _load_data(data, item2idx, idx_cnt, pad_idx, class_num):

    samplepack = Samplepack()
    samples = []
    now_id = 0
    sample = Sample()
    last_id = None
    click_items = []
    
    for s_id,item_id in zip(list(data['SessionId'].values),list(data['ItemId'].values)):
        if last_id is None:
            last_id = s_id
        if s_id != last_id:
            item_dixes = []
            for item in click_items:
                if item not in item2idx:
                    if idx_cnt == pad_idx:
                        idx_cnt += 1
                    item2idx[item] = idx_cnt
                    idx_cnt += 1
                item_dixes.append(item2idx[item])
            
            in_dixes = item_dixes[:-1]
            out_dixes = item_dixes[1:]
            sample.id = now_id
            sample.session_id = last_id
            sample.click_items = click_items
            sample.items_idxes = item_dixes
            sample.in_idxes = in_dixes
            sample.out_idxes = out_dixes
            samples.append(sample)
            sample = Sample()
            last_id =s_id
            click_items = []
            now_id += 1
        else:
            last_id = s_id
        click_items.append(item_id)
        # click_items = list(tmp_data[session_tmp_idx]['ItemId'])
    sample = Sample()
    item_dixes = []
    for item in click_items:
        if item not in item2idx:
            if idx_cnt == pad_idx:
                idx_cnt += 1
            item2idx[item] = idx_cnt
            idx_cnt += 1
        item_dixes.append(item2idx[item])
    in_dixes = item_dixes[:-1]
    out_dixes = item_dixes[1:]
    sample.id = now_id
    sample.session_id = last_id
    sample.click_items = click_items
    sample.items_idxes = item_dixes
    sample.in_idxes = in_dixes
    sample.out_idxes = out_dixes
    samples.append(sample)


    samplepack.samples = samples
    samplepack.init_id2sample()
    return samplepack, idx_cnt


