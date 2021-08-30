import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from joblib import Parallel, delayed
import time
import gc
import os
from processing_tools import *
from emulator import *

def main(emulation_mode=False):

    if emulation_mode:
        mlb = MLBEmulator(eval_start_day=20210501, eval_end_day=20210730)
    else:
        import mlb
    env = mlb.make_env() # initialize the environment
    iter_test = env.iter_test() # iterator which loops over each date in test set

    #read in players data
    players = pd.read_csv('/kaggle/input/mlb-player-digital-engagement-forecasting/players.csv')
    #read in teams data
    teams = pd.read_csv('/kaggle/input/mlb-player-digital-engagement-forecasting/teams.csv')
    #read in seasons
    seasons = pd.read_csv('/kaggle/input/mlb-player-digital-engagement-forecasting/seasons.csv')
    #read in awards_static
    awards_static = pd.read_csv('/kaggle/input/mlb-player-digital-engagement-forecasting/awards.csv')
    #Team clusters
    team_clusters = pd.read_csv('team_clusters.csv',index_col=0)
    #read in training_master
    training_master = pd.read_csv('training_master_updatedTrain_r3.csv')
    #Set path to Models
    models = ['Competition/model_target1_updatedTrainV3.json',
            'Competition/model_target2_updatedTrainV3.json',
            'Competition/model_target3_updatedTrainV3.json',
            'Competition/model_target4_updatedTrainV3.json',]

    for (test_df, sample_prediction_df) in iter_test:
        sample_prediction_df[['DatePrediction','playerId']] = sample_prediction_df['date_playerId'].str.split(pat='_',expand=True)
        sample_prediction_df.reset_index(inplace=True)
        Y_info = sample_prediction_df.loc[:,['date','playerId']]
        Y_info['date'] = pd.to_datetime(Y_info['date'],format='%Y%m%d')
        unpacked_test_dfs = unpack_data(test_df)
        training_master = create_X_master(unpacked_test_dfs, players, teams, seasons, awards_static,
                                        team_clusters,training_master, Y=Y_info)
        sample_prediction_df = make_predictions(training_master,sample_prediction_df,models)
        env.predict(sample_prediction_df)

if __name__ == '__main__':
    main()