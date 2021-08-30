#Notebook for cross-validation
import pandas as pd
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
import time
import numpy as np

#Due to memeory constraints, performing in batches.

def performCV_split1(data_path,dummy=False):
    #Read in training data
    print('Training first 2 splits...')
    training_master = pd.read_csv(data_path)
    if 'gamePk2' in list(training_master.columns):
        drop_cols = ['teamId','gamePk','gamePk2','engagementMetricsDate','playerId','dailyDataDate']
    else:
        drop_cols = ['teamId','gamePk','engagementMetricsDate','playerId','dailyDataDate']

    
    #Split training data into training and validation sets
    trainSet1 = training_master[(training_master['dailyDataDate'] < '2019-08-01')].drop(
            columns=drop_cols)
    valSet1 = training_master[(training_master['dailyDataDate'] >= '2019-08-01') & 
                              (training_master['dailyDataDate'] < '2019-09-01')].drop(
            columns=drop_cols)
    

    trainSet2 = training_master[(training_master['dailyDataDate'] < '2019-09-01')].drop(
            columns=drop_cols)
    valSet2 = training_master[(training_master['dailyDataDate'] >= '2019-09-01') & 
                              (training_master['dailyDataDate'] < '2019-10-01')].drop(
            columns=drop_cols) 

    trainSet3 = training_master[(training_master['dailyDataDate'] < '2020-08-01')].drop(
            columns=drop_cols)
    valSet3 = training_master[(training_master['dailyDataDate'] >= '2020-08-01') & 
                              (training_master['dailyDataDate'] < '2020-09-01')].drop(
            columns=drop_cols)

    trainSet4 = training_master[(training_master['dailyDataDate'] < '2020-09-01')].drop(
            columns=drop_cols)
    valSet4 = training_master[(training_master['dailyDataDate'] >= '2020-09-01') & 
                              (training_master['dailyDataDate'] < '2020-10-01')].drop(
            columns=drop_cols) 
    
    trainSets = [trainSet1,trainSet2,trainSet3,trainSet4]
    valSets = [valSet1,valSet2,valSet3,valSet4]
    del(trainSet1, valSet1, trainSet2, valSet2, trainSet3, valSet3, trainSet4, valSet4)
    training_master.drop(columns=drop_cols,inplace=True)        
    ncols = training_master.shape[1]
    del training_master
    
    #Split each set into features and labels     
    trainSets_features = []
    trainSets_labels = []
    valSets_features = []
    valSets_labels = []
    ind_features = list(range(ncols - 4))
    ind_label = ncols - 4
    
    #Process in chunks due to memory issues
    j = 0
    for trainSet,valSet in zip(trainSets,valSets):
        if j == 2:
            break
        trainSets_features.append(trainSet.iloc[:,ind_features])
        valSets_features.append(valSet.iloc[:,ind_features])
        trainSets_labels.append(trainSet.iloc[:,ind_label:])
        valSets_labels.append(valSet.iloc[:,ind_label:])       
        j+=1
    _ = trainSets.pop(0)
    _ = trainSets.pop(0)
    _ = valSets.pop(0)
    _ = valSets.pop(0)
    
    for trainSet,valSet in zip(trainSets,valSets):
        trainSets_features.append(trainSet.iloc[:,ind_features])
        valSets_features.append(valSet.iloc[:,ind_features])
        trainSets_labels.append(trainSet.iloc[:,ind_label:])
        valSets_labels.append(valSet.iloc[:,ind_label:])       
    
    del trainSets
    del valSets
    
    #Train dummy classifier to set as baseline
    mae_train = []
    mae_val = []    
    mae_train_raw = []
    mae_val_raw = []  
    #Perform in batches of 2 due to memory issues
    for i in range(2): #2 splits
        if dummy:
            regr = DummyRegressor(strategy="mean")
            regr.fit(trainSets_features[i],trainSets_labels[i])
            predicts_train = regr.predict(trainSets_features[i])
            predicts_val= regr.predict(valSets_features[i])
        else:
            predicts_train = []
            predicts_val = []
            #4 different models
            for j in range(4):
                regr = xgb.XGBRegressor()
                regr.fit(trainSets_features[i],trainSets_labels[i].iloc[:,j])
                predicts_train.append(regr.predict(trainSets_features[i]))
                predicts_val.append(regr.predict(valSets_features[i]))
            predicts_train = np.array(predicts_train).T
            predicts_val = np.array(predicts_val).T
        mae_train.append(mean_absolute_error(trainSets_labels[i], predicts_train))
        mae_val.append(mean_absolute_error(valSets_labels[i], predicts_val))   
        mae_train_raw.append(mean_absolute_error(trainSets_labels[i], predicts_train,multioutput='raw_values'))
        mae_val_raw.append(mean_absolute_error(valSets_labels[i], predicts_val,multioutput='raw_values'))  
    _ = trainSets_features.pop(0)
    _ = trainSets_features.pop(0)
    _ = valSets_features.pop(0)
    _ = valSets_features.pop(0)
    _ = trainSets_labels.pop(0)
    _ = trainSets_labels.pop(0)
    _ = valSets_labels.pop(0)
    _ = valSets_labels.pop(0)
    
    print('Training first 2 splits complete.')    
    print('Training next 2 splits...')
    for i in range(2): #2 splits
        if dummy:
            regr = DummyRegressor(strategy="mean")
            regr.fit(trainSets_features[i],trainSets_labels[i])
            predicts_train = regr.predict(trainSets_features[i])
            predicts_val= regr.predict(valSets_features[i])
        else:
            predicts_train = []
            predicts_val = []
            #4 different models
            for j in range(4):
                regr = xgb.XGBRegressor()
                regr.fit(trainSets_features[i],trainSets_labels[i].iloc[:,j])
                predicts_train.append(regr.predict(trainSets_features[i]))
                predicts_val.append(regr.predict(valSets_features[i]))
            predicts_train = np.array(predicts_train).T
            predicts_val = np.array(predicts_val).T
        mae_train.append(mean_absolute_error(trainSets_labels[i], predicts_train))
        mae_val.append(mean_absolute_error(valSets_labels[i], predicts_val))  
        mae_train_raw.append(mean_absolute_error(trainSets_labels[i], predicts_train,multioutput='raw_values'))
        mae_val_raw.append(mean_absolute_error(valSets_labels[i], predicts_val,multioutput='raw_values'))  
    del predicts_train
    del predicts_val
    del regr
    print('Training next 2 splits complete.')   
    return mae_val,mae_train,mae_train_raw,mae_val_raw

def performCV_split2(data_path,mae_train,mae_val,mae_train_raw,mae_val_raw,dummy=False):
    #Read in training data
    print('Training last split...')
    training_master = pd.read_csv(data_path)
    if 'gamePk2' in list(training_master.columns):
        drop_cols = ['teamId','gamePk','gamePk2','engagementMetricsDate','playerId','dailyDataDate']
    else:
        drop_cols = ['teamId','gamePk','engagementMetricsDate','playerId','dailyDataDate']
        
    #Split training data into training and validation sets

    trainSet5 = training_master[(training_master['dailyDataDate'] < '2021-06-01')].drop(
             columns=drop_cols)
    valSet5 = training_master[(training_master['dailyDataDate'] >= '2021-06-01')].drop(
             columns=drop_cols)  
    
    trainSets = [trainSet5]
    valSets = [valSet5]
    del(trainSet5, valSet5)
    training_master.drop(columns=drop_cols,inplace=True)        
    ncols = training_master.shape[1]
    del training_master
    
    #Split each set into features and labels     
    trainSets_features = []
    trainSets_labels = []
    valSets_features = []
    valSets_labels = []
    ind_features = list(range(ncols - 4))
    ind_label = ncols - 4
    
    #Process in chunks of 3 due to memory issues
    for trainSet,valSet in zip(trainSets,valSets):
        trainSets_features.append(trainSet.iloc[:,ind_features])
        valSets_features.append(valSet.iloc[:,ind_features])
        trainSets_labels.append(trainSet.iloc[:,ind_label:])
        valSets_labels.append(valSet.iloc[:,ind_label:])            
    
    del trainSet
    del valSet
    
    #Train dummy classifier to set as baseline 
    #Perform in batches of 2 due to memory issues
    for i in range(1): #2 splits
        if dummy:
            regr = DummyRegressor(strategy="mean")
            regr.fit(trainSets_features[i],trainSets_labels[i])
            predicts_train = regr.predict(trainSets_features[i])
            predicts_val= regr.predict(valSets_features[i])
        else:
            predicts_train = []
            predicts_val = []
            #4 different models
            for j in range(4):
                regr = xgb.XGBRegressor()
                regr.fit(trainSets_features[i],trainSets_labels[i].iloc[:,j])
                predicts_train.append(regr.predict(trainSets_features[i]))
                predicts_val.append(regr.predict(valSets_features[i]))
            predicts_train = np.array(predicts_train).T
            predicts_val = np.array(predicts_val).T
        mae_train.append(mean_absolute_error(trainSets_labels[i], predicts_train))
        mae_val.append(mean_absolute_error(valSets_labels[i], predicts_val))   
        mae_train_raw.append(mean_absolute_error(trainSets_labels[i], predicts_train,multioutput='raw_values'))
        mae_val_raw.append(mean_absolute_error(valSets_labels[i], predicts_val,multioutput='raw_values'))  
    print('Training last split complete.')
    return mae_val,mae_train,mae_train_raw,mae_val_raw

def performCV(data_path,dummy=False):
    start = time.time()
    mae_val, mae_train,mae_train_raw,mae_val_raw = performCV_split1(data_path,dummy)    
    mae_val,mae_train,mae_train_raw,mae_val_raw = performCV_split2(data_path,mae_train,mae_val,mae_train_raw,mae_val_raw,dummy)
    print('Training Set MAE overall average: {:.3f}'.format(np.average(mae_train)))
    print('Validation Set MAE: {:.3f}'.format(np.average(mae_val)))
    print('Training Set MAE for each split: ',mae_train)
    print('Validation Set MAE for each split: ',mae_val)
    print('Training Set MAE for each split, multiouput: ',mae_train_raw)
    print('Validation Set MAE for each split, multiouput: ',mae_val_raw)
    end = time.time()
    print('Elapsed time: ',end - start)
    return np.average(mae_val),np.average(mae_train),mae_val,mae_train,mae_train_raw,mae_val_raw