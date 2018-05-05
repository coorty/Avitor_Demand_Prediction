# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:49:31 2018

@author: coorty

可以加入的特征:
    1) Title和Description中单词的数量;
    2) 按照user_id进行分组计数;
    3) city特征的mean encode特征;

"""
import numpy as np
import pandas as pd
import gc
from time import time, strftime
from tqdm import tqdm
from contextlib import contextmanager
from operator import itemgetter

import xgboost as xgb
import lightgbm as lgb
from nltk.corpus import stopwords

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer

@contextmanager
def timer(name):
    """ Counting running time
    - name: to be displayed
    """
    print('-'*30)
    print('[{}] Begin: {}'.format(name, strftime('%Y-%m-%d %H:%M:%S')))
    t0 = time()
    yield
    print('[{}] End  : {}'.format(name, strftime('%Y-%m-%d %H:%M:%S')))   
    print(f'[{name}] done cost {time() - t0:.1f} s')
    
def make_field_pipeline(field: str, *vec) -> Pipeline:
    """ Make Pipeline with refer to field : `field`, and some transform functions: `*vec`
    Input:
        - field: a data field
        - *vec: a sequence transformance functions
    """
    return make_pipeline(FunctionTransformer(itemgetter(field), validate=False), *vec)

def rmse(y_pred, y_true):
    """ rmse
    """
    n = len(y_pred)
    return np.sqrt(np.sum((y_pred - y_true)**2)/n)

def data_preprocess(data):
    """ Perform data procession
    
    """
    field_list = []
    
    ### On `param_1`(76% is null), `param_2`(86% is null), `param_3` (89% is null)
    # This three params are text-type, and most of them are empty.
    # Concat `description`+`title`+`param_1`+'param_2'+`param_3`+`city`
    data['description'].fillna(' ', inplace=True)
    data['param_1'].fillna(' ', inplace=True)
    data['param_2'].fillna(' ', inplace=True)
    data['param_3'].fillna(' ', inplace=True)
    data['text'] = data['description']+' '+data['title']+' '+data['param_1']+' '+\
                   data['param_2']+' '+data['param_3']+' '+data['city']
                           
    ### On `description` and `title` (text type)
#    data['description'].fillna('', inplace=True)
#    data['title'].fillna('', inplace=True)
    
    ### On `activation_date` (categorical type)
    data['month']     = data['activation_date'].dt.month.astype('int')
    data['day']       = data['activation_date'].dt.day.astype('int')
    data['dayofweek'] = data['activation_date'].dt.dayofweek.astype('int')
    data['week']      = data['activation_date'].dt.week.astype('int')
    field_list.extend(['month', 'day', 'dayofweek', 'week'])
    
    # Label categorical features
    le = LabelEncoder()
    
    ### On `city`      (categorical type, 1752-unique-values on whole dataset, no null)
#    data['city'] = le.fit_transform(data['city'])
    
    ### OK On `region`    (categorical type, 28-unique-values on whloe dataset, no null)
    data['region'] = le.fit_transform(data['region'])
    data['region'] = data['region'].astype('int')
    field_list.append('region')
    
    ### OK On `user_type` (categorical type, 3-unique-values on whloe dataset, no null)
    data['user_type'] = le.fit_transform(data['user_type'])
    data['user_type'] = data['user_type'].astype('int')
    field_list.append('user_type')
    
    ### OK On `category_name` (categorical type, 47-unique-values on whloe dataset, no null)
    data['category_name'] = le.fit_transform(data['category_name'])
    data['category_name'] = data['category_name'].astype('int')
    field_list.append('category_name')
    
    ### OK On `parent_category_name` (categorical type, 9-unique-values on whloe dataset, no null)
    data['parent_category_name'] = le.fit_transform(data['parent_category_name'])
    data['parent_category_name'] = data['parent_category_name'].astype('int')
    field_list.append('parent_category_name')
    
    ### On `item_seq_number` (categorical numerical type, 1,2,...; 33947-unique-value)
#    data.loc[data['item_seq_number']>100000, 'item_seq_number'] = 100000
#    data['item_seq_number'] = data['item_seq_number'].astype('category')
    
    ### On `image` (Id code of image. Ties to a jpg file in train_jpg. Not every ad has an image.)
    data['has_image'] = data['image'].notnull().astype('int')
    field_list.append('has_image')
    
    ### On `price` (numerical type, mean=307410.4, has some null)
    # Compute average price of each group and fill the null
    data['price'] = data[['city','region','user_type','price']].groupby(['city','region','user_type']).\
                                            transform(lambda x: x.fillna(x.mean()))
    data['price'] = np.log1p(data['price'])
    field_list.append('price')
    
    ### Not use `image_top_1` (Avito's classification code for the image.)
    
    return field_list




def lgb_train(train_x, val_x, train_y, val_y):
    """ train a lightgbm regression model using `train_x` and `train_y`
    """
    print('Train lightgbm...')
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.09,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "verbosity" : -1  }
    
    # train and validation set
    lgtrain = lgb.Dataset(train_x, label=train_y)
    lgval   = lgb.Dataset(val_x, label=val_y)
    
    model = lgb.train(params, lgtrain, 3000, valid_sets=[lgval], 
                      verbose_eval=1)
    
    train_pred = model.predict(data=train_x, num_iteration=model.best_iteration)
    train_pred[train_pred < 0] = 0
    train_pred[train_pred > 1] = 1
    
    val_pred   = model.predict(data=val_x, num_iteration=model.best_iteration)
    val_pred[val_pred < 0] = 0
    val_pred[val_pred > 1] = 1
    
    print('rmse on trainset is: {:.5f}'.format(rmse(train_pred, train_y)))
    print('rmse on validation set is: {:.5f}'.format(rmse(val_pred, val_y)))
    
    return model
    

def xgboost_train(train_x, val_x, train_y, val_y):
    print('Train xgboost...')
    params = {
            'n_jobs': -1,
            'objective': 'reg:linear',
            'learning_rate': 0.3,
            'subsample': .70,
            'max_depth': 6
            }
    
    watchlist = [(xgb.DMatrix(train_x, train_y),'train'),(xgb.DMatrix(val_x, val_y),'valid')]
    
    evals_result = {}
    model = xgb.train(params=params, dtrain=xgb.DMatrix(train_x, train_y), num_boost_round=400,
                      evals=watchlist, evals_result=evals_result, verbose_eval=1)
        
    train_pred = model.predict(data=xgb.DMatrix(train_x))
    train_pred[train_pred < 0] = 0
    train_pred[train_pred > 1] = 1
    
    val_pred   = model.predict(data=xgb.DMatrix(val_x))
    val_pred[val_pred < 0] = 0
    val_pred[val_pred > 1] = 1
    
    print('rmse on trainset is: {:.5f}'.format(rmse(train_pred, train_y)))
    print('rmse on validation set is: {:.5f}'.format(rmse(val_pred, val_y)))
    
    return model
    


if __name__ == '__main__':
    model_name = 'xgboost'
    stop_words = stopwords.words('russian')
    
    with timer('Load data'):
        train = pd.read_csv('../data/train.csv.zip', compression='zip', parse_dates=['activation_date'])
        test  = pd.read_csv('../data/test.csv.zip', compression='zip', parse_dates=['activation_date'])
        n_train = len(train)
        merge = pd.concat((train, test), ignore_index=True, axis=0)
        other_fields = data_preprocess(merge)
    
    with timer('Extract text features'):
        vectorizer = make_union(
                make_field_pipeline('text', 
                                    TfidfVectorizer(max_features=50000,stop_words=stop_words, ngram_range=(1,2), max_df=.95, min_df=2),
                                    TruncatedSVD(n_components=50, algorithm='arpack')),
                make_field_pipeline('text',
                                    CountVectorizer(max_features=50000,stop_words=stop_words, ngram_range=(1,2), max_df=.95, min_df=2),
                                    LatentDirichletAllocation(n_components=50)),              
                make_field_pipeline(other_fields)
                )
                
        all_feats = vectorizer.fit_transform(merge)
        del merge
        gc.collect()
        
    # train & test features
    index = np.squeeze(np.argwhere(fi>=400))
    train_x, train_y = all_feats[:n_train, index], train['deal_probability'].values
    test_x = all_feats[n_train:, index]
                
    with timer('Model train'):
        if model_name == 'xgboost':
            model = xgboost_train(*train_test_split(train_x, train_y, 
                                                        test_size=0.10, random_state=42))
        elif model_name == 'lightgbm':
            model = lgb_train(*train_test_split(train_x, train_y, 
                                                    test_size=0.10, random_state=42))            
        else:
            pass
        
        
    with timer('Model test'):        
        test_pred = model.predict(data=test_x, num_iteration=model.best_iteration)
        test_pred[test_pred<0] = 0
        test_pred[test_pred>1] = 1
        
    with timer('Write results'):
        submission = pd.DataFrame(test['item_id'])
        submission['deal_probability'] = test_pred
        submission.to_csv('avito_submission_'+strftime('%m_%d_%H_%M')+'.csv', index=False)

    print('Done !')
    
    
    
"""
------------------------------
[Load data] Begin: 2018-05-03 17:19:30
[Load data] End  : 2018-05-03 17:20:30
[Load data] done cost 59.1 s
------------------------------
[Extract text features] Begin: 2018-05-03 17:20:30
[Extract text features] End  : 2018-05-04 02:54:29
[Extract text features] done cost 34439.7 s
------------------------------
[Model train] Begin: 2018-05-04 02:54:29
rmse on trainset is: 0.19068
rmse on validation set is: 0.22531
[Model train] End  : 2018-05-04 03:03:43
[Model train] done cost 553.8 s
------------------------------
[Model test] Begin: 2018-05-04 03:03:43
[Model test] End  : 2018-05-04 03:03:59
[Model test] done cost 15.6 s
------------------------------
[Write results] Begin: 2018-05-04 03:03:59
[Write results] End  : 2018-05-04 03:04:01
[Write results] done cost 1.8 s
Done !
"""
    






    
    
    # 0.2337
    # 0.2310
   
    
    



