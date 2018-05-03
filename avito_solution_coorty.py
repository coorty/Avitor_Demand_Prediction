# -*- coding: utf-8 -*-
"""
# Author: coorty

This is a coarse solution for `Kaggle: Avito Demand Prediction Challenge(Predict demand for an online classified ad)(https://www.kaggle.com/c/avito-demand-prediction)`

The rmse is 0.2263.


------------
MIT License.

"""
import numpy as np
import pandas as pd
import gc
from time import time, strftime
from tqdm import tqdm
from contextlib import contextmanager
from operator import itemgetter

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

def data_preprocess(data):
    """ Perform data procession
    
    """
    ### On `description` and `title` (text type)
    data['description'].fillna('', inplace=True)
    data['title'].fillna('', inplace=True)
    
    ### On `activation_date` (categorical type)
    data['month'] = data['activation_date'].dt.month.astype('category')
    data['day'] = data['activation_date'].dt.day.astype('category')
    data['dayofweek'] = data['activation_date'].dt.dayofweek.astype('category')
    
    #
    le = LabelEncoder()
    
    ### On `city`      (categorical type, 1752-unique-values on whole dataset, no null)
    data['city'] = le.fit_transform(data['city'])
    
    ### On `region`    (categorical type, 28-unique-values on whloe dataset, no null)
    data['region'] = le.fit_transform(data['region'])
    
    ### On `user_type` (categorical type, 3-unique-values on whloe dataset, no null)
    data['user_type'] = le.fit_transform(data['user_type'])
    
    ### On `category_name` (categorical type, 47-unique-values on whloe dataset, no null)
    data['category_name'] = le.fit_transform(data['category_name'])
    
    ### On `parent_category_name` (categorical type, 9-unique-values on whloe dataset, no null)
    data['parent_category_name'] = le.fit_transform(data['parent_category_name'])
    
    ### On `item_seq_number` (categorical numerical type, 1,2,...,)
    data.loc[data['item_seq_number']>100000, 'item_seq_number'] = 100000
    data['item_seq_number'] = data['item_seq_number'].astype('category')
    
    ### On `image` (Id code of image. Ties to a jpg file in train_jpg. Not every ad has an image.)
    data['has_image'] = data['image'].notnull().astype('int')
    
    ### On `price` (numerical type, mean=307410.4, has some null)
    # Compute average price of each groups and fill the null
    data['price'] = data[['city','region','user_type','price']].groupby(['city','region','user_type']).\
                                            transform(lambda x: x.fillna(x.mean()))

    ### Not use `image_top_1` (Avito's classification code for the image.)
    ### Not use `param_1`(76% is null), `param_2`(86% is null), `param_3` (89% is null)

    return ['month', 'day', 'dayofweek', 'city', 'region', 'user_type', 'category_name', 
            'parent_category_name', 'item_seq_number', 'has_image', 'price']

def rmse(y_pred, y_true):
    n = len(y_pred)
    return np.sqrt(np.sum((y_pred - y_true)**2)/n)

def lgb_train(train_x, val_x, train_y, val_y):
    """ train a lightgbm regression model using `train_x` and `train_y`
    """
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.09,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "verbosity" : -1  }
    
    lgtrain = lgb.Dataset(train_x, label=train_y)
    lgval   = lgb.Dataset(val_x, label=val_y)
    
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, valid_sets=[lgval], 
                      verbose_eval=20, evals_result=evals_result)
    
    train_pred = model.predict(data=train_x, num_iteration=model.best_iteration)
    val_pred   = model.predict(data=val_x, num_iteration=model.best_iteration)
    print('rmse on trainset is: {:.5f}'.format(rmse(train_pred, train_y)))
    print('rmse on validation set is: {:.5f}'.format(rmse(val_pred, val_y)))
    
    return model, evals_result
    


if __name__ == '__main__':
    
    stop_words = stopwords.words('russian')
    
    with timer('Load data'):
        train = pd.read_csv('./data/train.csv.zip', compression='zip', parse_dates=['activation_date'])
        test  = pd.read_csv('./data/test.csv.zip', compression='zip', parse_dates=['activation_date'])
        n_train = len(train)
        merge = pd.concat((train, test), ignore_index=True, axis=0)
        other_fields = data_preprocess(merge)
    
    with timer('Extract text features'):
        vectorizer = make_union(
                make_field_pipeline('description', 
                                    TfidfVectorizer(max_features=50000,stop_words=stop_words, ngram_range=(1,2)),
                                    TruncatedSVD(n_components=5, algorithm='arpack')),
                make_field_pipeline('title', 
                                    TfidfVectorizer(max_features=10000,stop_words=stop_words, ngram_range=(1,2)),
                                    TruncatedSVD(n_components=5, algorithm='arpack')),
                make_field_pipeline('description',
                                    CountVectorizer(max_features=50000,stop_words=stop_words, ngram_range=(1,2), max_df=.95, min_df=2),
                                    LatentDirichletAllocation(n_components=5)),
                make_field_pipeline('title',
                                    CountVectorizer(max_features=10000,stop_words=stop_words, ngram_range=(1,2), max_df=.95, min_df=2),
                                    LatentDirichletAllocation(n_components=5)),                 
                make_field_pipeline(other_fields)
                )
                
        all_feats = vectorizer.fit_transform(merge)
        del merge
        gc.collect
        
    # train & test features   
    train_x, train_y = all_feats[:n_train, :], train['deal_probability']
    test_x = all_feats[n_train:, :]
                
    with timer('Model train'):
        lgb_model, evals_result = lgb_train(*train_test_split(train_x, train_y, test_size=0.20, random_state=42))
        
    with timer('Model test'):        
        test_pred = lgb_model.predict(data=test_x, num_iteration=lgb_model.best_iteration)
        test_pred[test_pred<0] = 0
        test_pred[test_pred>1] = 1
        
    with timer('Write results'):
        submission = pd.DataFrame(test['item_id'])
        submission['deal_probability'] = test_pred
        submission.to_csv('avito_submission_'+strftime('%m_%d_%H_%M')+'.csv', index=False)

    print('Done !')
    
    
    # 0.2337
                
#                on_field('text', Tfidf(max_features=100, token_pattern='\w+', ngram_range=(1,2))),
#                on_field(['shipping', 'item_condition_id'],
#                         FunctionTransformer(to_records, validate=False), DictVectorizer()),
#                n_jobs=1)
#
#
#
#
#
#
#
#russianStopwords = stopwords.words('russian')
#
#### Load train and test data
#
#
#
#nTrain, nTest = len(train), len(test)
#
#"""
#Process `description` and `title` column
#"""
#print('[{}] Process `description` and `title` column'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
#train['description'].fillna(' ', inplace=True)
#test['description'].fillna(' ', inplace=True)
#
### Extracting Tfidf matrix
#descVectorizer  = TfidfVectorizer(max_features=50000, stop_words=russianStopwords)
#titleVectorizer = TfidfVectorizer(max_features=50000, stop_words=russianStopwords)
#
## 2011862*50000 sparse matrix type '<class 'numpy.float64'>'
#descTfidfMat = descVectorizer.fit_transform(pd.concat((train['description'], test['description']),
#                                                      ignore_index=True))
#
## 2011862*50000 sparse matrix type '<class 'numpy.float64'>'
#titleTfidfMat = descVectorizer.fit_transform(pd.concat((train['title'], test['title']),
#                                                      ignore_index=True))
### Extracting SVD matrix
#descSVD  = TruncatedSVD(n_components=3, algorithm='arpack')
#titleSVD = TruncatedSVD(n_components=3, algorithm='arpack')
#
## 2011862 * 3 'numpy.ndarray'
#descSVDMat = descSVD.fit_transform(descTfidfMat)
#
## 2011862 * 3 'numpy.ndarray'
#titleSVDMat = titleSVD.fit_transform(titleTfidfMat)
#
#trainTextFeat = pd.DataFrame(np.hstack((descSVDMat[:nTrain,:], titleSVDMat[:nTrain,:])),
#                        columns=['descSVD_'+str(i+1) for i in range(3)]+['titleSVD_'+str(i+1) for i in range(3)])
#
#testTextFeat  = pd.DataFrame(np.hstack((descSVDMat[nTrain:,:], titleSVDMat[nTrain:,:])),
#                        columns=['descSVD_'+str(i+1) for i in range(3)]+['titleSVD_'+str(i+1) for i in range(3)])
#
#del descTfidfMat, titleTfidfMat, descSVDMat, titleSVDMat
#gc.collect()
#
#
#"""
#Process category columns
#
#类别特征:
#    - region: no null; 28 unique-value;   most frequent: 186514(9.27%)
#    - city  : no null; 1752 unique-value; most frequent: 85993(4.27%)
#    - parent_category_name: no null; 9 unique-value; most frequent: 914200(45.44%)
#    - category_name: no null; 47 unique-value; most frequent: 367649(18.27%)
#    - user_type: no null; 3 unique-value; most frequent: 1433965(71.28%)
#"""
#print('[{}] Process category columns'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
#categoryVars = ['region', 'city', 'parent_category_name', 'category_name', 'user_type']
#for col in tqdm(categoryVars):
#    le = LabelEncoder()
#    le.fit(pd.concat((train[col], test[col]), ignore_index=True))
#    train[col] = le.transform(train[col])
#    test[col]  = le.transform(test[col])
#    
#    
#"""
#Process other columns
#    - activation_date: 转换成为星期
#"""
#print('[{}] Process other columns'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
#otherVars = ['activation_date', 'price']
#train['activation_date'] = train['activation_date'].dt.weekday
#test['activation_date']  = test['activation_date'].dt.weekday
#    
#
#"""
#Merge all features
#"""
#def rmsl(y_true, y_pred):
#    n = len(y_true)
#    rmse = np.sqrt(np.sum((y_true - y_pred)**2)/n)
#    return rmse
#
#
#def lgb_train(train_x, val_x, train_y, val_y):
#    params = {'objective': 'regression',
#              'metric': 'rmse',
#              'num_leaves': 40,
#              'learning_rate': .09,
#              'bagging_fraction': .7,
#              'feature_fraction': .7,
#              'bagging_frequency': 5,
#              'bagging_seed': 2018,
#              'verbosity': -1}
#    
#    trainLGB = lgb.Dataset(data=train_x, label=train_y)
#    valLGB   = lgb.Dataset(data=val_x, label=val_y)
#    
#    # Model training
#    evals_result = {}
#    lgbModel = lgb.train(params=params, train_set=trainLGB, num_boost_round=5000, valid_sets=[valLGB],
#                         verbose_eval=20, evals_result=evals_result)
#    
#    return lgbModel, evals_result
#    
#    # Prediction
##    testPred = lgbModel.predict(data=test_x, num_iteration=lgbModel.best_iteration)
##    return lgbModel, evals_result, testPred
#    
#
#### Concat all features
#trainX = pd.concat((trainTextFeat, train[categoryVars], train[otherVars]), axis=1, ignore_index=True)
#testX  = pd.concat((testTextFeat,  test[categoryVars],  test[otherVars]), axis=1, ignore_index=True)
#
#### Model training
#lgbModel, evals_result = lgb_train(*train_test_split(trainX.values, train.deal_probability, 
#                                                     test_size=0.20, random_state=42))
#
#### Model testing
#testPred = lgbModel.predict(data=testX, num_iteration=lgbModel.best_iteration)
#testPred[testPred>1] = 1    
#testPred[testPred<0] = 0    
#    
#trainPred = lgbModel.predict(data=trainX, num_iteration=lgbModel.best_iteration)
#trainPred[trainPred>1] = 1    
#trainPred[trainPred<0] = 0   
#print('trainset rmsl: ', str(rmsl(train.deal_probability, trainPred)))    
#    
#submission = pd.DataFrame(testPred, columns=['deal_probability'], index=test.item_id)    
#submission.to_csv('Avito_first_submission.csv', index=True, index_label='item_id')   
    
    
    
    



