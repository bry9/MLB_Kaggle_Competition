import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from joblib import Parallel, delayed
import time
import datetime
import gc
import os
from xgboost import XGBRegressor
import xgboost as xgb

#Various helper functions to load / check / process the dataframes
#=======================================================================================
#Unpacked training data dfs and assigning to variables
def unpack_data(test_df):
    unpacked_test_dfs = []
    daily_data_nested_df_names = test_df.columns.values.tolist()
    if 'date' in daily_data_nested_df_names:
        flag = True
        daily_data_nested_df_names.remove('date')
    for df_name in daily_data_nested_df_names:
        if flag:
            date_nested_table = test_df.rename(columns={'date':'dailyDataDate'})[['dailyDataDate',df_name]]
        else:
            try:
                date_nested_table = test_df.reset_index().rename(columns={'index':'dailyDataDate'})[['dailyDataDate',df_name]]
            except:
                date_nested_table = test_df.reset_index().rename(columns={'date':'dailyDataDate'})[['dailyDataDate',df_name]]
        date_nested_table['dailyDataDate'] = pd.to_datetime(date_nested_table['dailyDataDate'], format = "%Y%m%d")

        date_nested_table = (date_nested_table[
              ~pd.isna(date_nested_table[df_name])
              ].
              reset_index(drop = True)
              )
        daily_dfs_collection = []
        if not date_nested_table.empty:
            for date_index, date_row in date_nested_table.iterrows():
                daily_df = pd.read_json(date_row[df_name])
                daily_df['dailyDataDate'] = date_row['dailyDataDate']                     
                daily_dfs_collection = daily_dfs_collection + [daily_df]

            unnested_table = (pd.concat(daily_dfs_collection,
                  ignore_index = True).
                  # Set and reset index to move 'dailyDataDate' to front of df
                  set_index('dailyDataDate').
                  reset_index()
                  )    # Creates 1 pandas df per unnested df from daily data read in, with same name
            unpacked_test_dfs.append(unnested_table)
        else:
            unpacked_test_dfs.append(date_nested_table)
            unnested_table = []

        # Clean up tables and collection of daily data frames for this df
        del(date_nested_table, daily_dfs_collection, unnested_table)    
    return unpacked_test_dfs

def assign_vars(unpacked_dfs, Y = None):
    if Y is not None:
        if len(unpacked_dfs) == 10:
            games = unpacked_dfs[0]                          
            rosters = unpacked_dfs[1]                       
            playerBoxScores = unpacked_dfs[2]              
            teamBoxScores = unpacked_dfs[3]                 
            transactions = unpacked_dfs[4]                   
            standings = unpacked_dfs[5]                      
            awards = unpacked_dfs[6]                         
            events = unpacked_dfs[7]                         
            playerTwitterFollowers = unpacked_dfs[8]         
            teamTwitterFollowers = unpacked_dfs[9]               
        elif len(unpacked_dfs) == 11:
            _ = unpacked_dfs[0]
            games = unpacked_dfs[1]                          
            rosters = unpacked_dfs[2]                       
            playerBoxScores = unpacked_dfs[3]              
            teamBoxScores = unpacked_dfs[4]                 
            transactions = unpacked_dfs[5]                   
            standings = unpacked_dfs[6]                      
            awards = unpacked_dfs[7]                         
            events = unpacked_dfs[8]                         
            playerTwitterFollowers = unpacked_dfs[9]         
            teamTwitterFollowers = unpacked_dfs[10]    

        return games, rosters, playerBoxScores, teamBoxScores, transactions, \
                   standings, awards, events, playerTwitterFollowers, teamTwitterFollowers         
    else:
        nextDayPlayerEngagements = unpacked_dfs[0]
        games = unpacked_dfs[1]                          
        rosters = unpacked_dfs[2]                       
        playerBoxScores = unpacked_dfs[3]              
        teamBoxScores = unpacked_dfs[4]                 
        transactions = unpacked_dfs[5]                   
        standings = unpacked_dfs[6]                      
        awards = unpacked_dfs[7]                         
        events = unpacked_dfs[8]                         
        playerTwitterFollowers = unpacked_dfs[9]         
        teamTwitterFollowers = unpacked_dfs[10]    

        return nextDayPlayerEngagements, games, rosters, playerBoxScores, teamBoxScores, \
               transactions, standings, awards, events, playerTwitterFollowers, teamTwitterFollowers

#Rosters

def bin_status(x):
    if x not in ['a','d10','d60','d7','rm']:
        return 'other'
    elif x in ['d10','d60','d7']:
        return 'dl'
    else:
        return x
    
def bin_teams(x, team_ids):
    if x not in team_ids:
        return 'other'
    else:
        return x
    
def make_rosters(df, date_features, float_features, bool_features, cat_features,
                 player_ids, team_ids):

    all_features = date_features + float_features + bool_features + cat_features
    
    #Define exepcted roster statuses
    expected_statuses = ['statusCode_a','statusCode_dl',
                             'statusCode_other','statusCode_rm']

    #Main processing
    if df.empty: #Create empty dataframe with expected features
        X = pd.DataFrame(columns=all_features) 
        
        #Drop statusCode in favor for one hot encoded version
        X.drop(columns=['statusCode'],inplace=True)     
        X[expected_statuses] = None              
    else:
        print('Rosters df is not null')
        #Extract features of interest from dataframe
        X = df.loc[:,all_features]

        # Set dtypes
        X = X.astype({name: np.float32 for name in float_features})
        X = X.astype({name: str for name in cat_features}) 

        #Strip and lower case strings
        for name in cat_features:
            X.loc[:,name] = X.loc[:,name].str.strip().str.lower()

        #bin teams to major league and non-major league
        X['teamId'] = X['teamId'].apply(lambda x:bin_teams(x,team_ids))
        #do not oneHotEncode teams yet, as they will be used to merge dataframes later on. 

        #Bin player roster status
        X['statusCode'] = X['statusCode'].apply(lambda x:bin_status(x))

        #one hot encode player roster status
        X = pd.get_dummies(X,columns=['statusCode'])

        #If instance of expected features are not present, add in.
        for status in expected_statuses:
            if status not in X.columns:
                X[status] = 0    

        #Filter to players that are in the player ids of interest
        X = X[X['playerId'].isin(player_ids)]
    
    return X 

#=======================================================================================
#Player Info
def bin_countries(x):
    if x == 'usa':
        return 1
    else:
        return 0
    
def make_playerInfo(df, date_features, float_features, bool_features ,cat_features,
                    player_ids, dailyDataDates):
    
    all_features = date_features + float_features + bool_features + cat_features
    
    #No need to check if playerInfo df is empty, since it is a static file and always available
    
    #Extract features of interest from dataframe
    X = df.loc[:,all_features]
    
    # Set dtypes
    X = X.astype({name: np.float32 for name in float_features})
    X = X.astype({name: str for name in cat_features}) 
    
    #Strip and lower case strings
    for name in cat_features:
        X.loc[:,name] = X.loc[:,name].str.strip().str.lower()
        
    X = X[X['playerId'].isin(player_ids)]    
    
    #Add new feature Born in US by processing birthCountry, then drop birthCountry
    X['BornInUS?'] = X['birthCountry'].apply(lambda x: bin_countries(x))
    X.drop(columns = ['birthCountry'],inplace=True)
    
    #Sanity check player weights and height
    #Take absoluve value of height, and ensure it is within 4 and 8 ft. 
    #Perform similar clipping for weight to ensure reasonable values
    X['heightInches'] = X['heightInches'].abs().clip(48,100)
    X['weight'] = X['weight'].abs().clip(100,400)

    #Assign date time to mlbDebutDate
    X['mlbDebutDate'] = pd.to_datetime(X['mlbDebutDate'], format = "%Y-%m-%d")
    
    #One hot encode primary postion code
    X = pd.get_dummies(X,columns=['primaryPositionCode'])
    
    #If instance of expected position is not present, add in
    expected_positions = ['primaryPositionCode_1',        
                          'primaryPositionCode_10',       
                          'primaryPositionCode_2',        
                          'primaryPositionCode_3',        
                          'primaryPositionCode_4',        
                          'primaryPositionCode_5',        
                          'primaryPositionCode_6',        
                          'primaryPositionCode_7',        
                          'primaryPositionCode_8',        
                          'primaryPositionCode_9',        
                          'primaryPositionCode_i',        
                          'primaryPositionCode_o']  
    for position in expected_positions:
        if position not in X.columns:
            X[position] = 0
            
    #Add in daily data date
    daily_dfs_collection = []
    for date in dailyDataDates:
        daily_df = X.copy()
        daily_df['dailyDataDate'] = date     
        daily_dfs_collection.append(daily_df)

    X_playerInfo = (pd.concat(daily_dfs_collection,
          ignore_index = True).
          # Set and reset index to move 'dailyDataDate' to front of df
          set_index('dailyDataDate').sort_index().
          reset_index()
          )    
    
    #Convert mlb debut date into days relative to daily data date
    X_playerInfo['mlbDebutDate_DaysRelative'] = (X_playerInfo['mlbDebutDate'] 
                                                 - X_playerInfo['dailyDataDate']).dt.days
    X_playerInfo.drop(columns = ['mlbDebutDate'],inplace=True)
    #Reorder to have dailyDataDate, playerId, and mlbDebute date as first 3 columns
    X_playerInfo = X_playerInfo.iloc[:,[0,3,17,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16]]
    
    return X_playerInfo

#=======================================================================================
#Merge rosters and playerInfo, and process NaNs
def merge_rosters_playerInfo_process_NaNs(X_rosters, X_playerInfo, training_master=None):
    #Check if X_rosters is empty. No need to check X_playerInfo since this will always exist.
    if X_rosters.empty:
        X_master = X_playerInfo
        #Add in additional columns that would've been added by merge, and fill in with nan
        X_roster_cols = list(X_rosters.columns)
        X_roster_cols.remove('dailyDataDate')
        X_roster_cols.remove('playerId')
        X_master[X_roster_cols] = np.nan
    else:
        X_master = X_playerInfo.merge(X_rosters, how='outer',
                                    left_on = ['dailyDataDate','playerId'],
                                    right_on = ['dailyDataDate','playerId'])        
    if training_master is not None:
        #Utilize previous day(s) data to fill in NaNs
        cols = list(X_master.columns)
        X_master = pd.concat([training_master,X_master],axis=0,ignore_index=True)
        inds_orig_training = training_master.index
        X_master = X_master.loc[:,cols].sort_values(by=['playerId','dailyDataDate',])
        X_master[['teamId','statusCode_a','statusCode_dl','statusCode_other','statusCode_rm']] \
                         = X_master.groupby('playerId')[['teamId','statusCode_a',
                                      'statusCode_dl','statusCode_other',
                                      'statusCode_rm']].fillna(method='ffill').fillna(method='bfill')
        X_master.drop(index=inds_orig_training,inplace=True)
    else:
        #handle NaNs in teamId / status Codes
        #If any teamId's / statusCodes are Nan, forward fill / backfill them with last known team. 
        #Sort by playerId and dailydatadate
        X_master.sort_values(by=['playerId','dailyDataDate'],inplace=True)
        X_master[['teamId','statusCode_a','statusCode_dl','statusCode_other','statusCode_rm']] \
                  = X_master.groupby('playerId')[['teamId','statusCode_a','statusCode_dl','statusCode_other',
                                                  'statusCode_rm']].fillna(method='ffill').fillna(method='bfill')

    #Handle Nans in mlbDebutDate (they are players that have not yet made it to the majors)...
    #Assume they will make debut on June 1, 2022
    debut_date = datetime.datetime(2022, 6, 1)
    d = {pd.Timestamp(date): (debut_date - pd.Timestamp(date)).days
                    for date in X_master['dailyDataDate'].unique()}
    X_master['mlbDebutDate_DaysRelative'] = X_master['mlbDebutDate_DaysRelative'] \
                                                .fillna(X_master['dailyDataDate'].map(d))
    #Reset indicies
    X_master.reset_index(drop=True,inplace=True)
        
    return X_master
#=======================================================================================
#Player Box Scores
#Helper function for aggregation of position Names
pos = ['catcher','first base','outfielder','pitcher','second base','shortstop','third base']

def agg_positions(arg,pos = pos):
    if len(arg) == 1:
        return arg
    elif len(arg) == 2:
        if arg.iloc[0] in pos:
            return arg.iloc[0]
        elif arg.iloc[1] in pos:
            return arg.iloc[1]
        else:
            return min(arg)
    else:
        return min(arg)
    
def gamePk(arg):
    if arg.nunique() == 1:
        return arg.unique()[0]
    else:
        return min(arg)

    
def gamePk2(arg):
    if arg.nunique() == 1:
        return -1
    else:
        return max(arg)
    
def make_playerBoxScores(df, date_features, float_features, bool_features,
                         cat_features, player_ids, team_ids):                   
                        
    all_features = date_features + float_features + bool_features + cat_features
    #Define expected positions, to be used later in if-else block
    positions = ['catcher',
                'designated hitter',
                'first base',
                'outfielder',
                'pinch hitter',
                'pinch runner',
                'pitcher',
                'second base',
                'shortstop',
                'third base']    
    expected_positions = ['positionName_playedInGame_' + position for position in positions]   
    
    #Main processing
    if df.empty: #Create empty dataframe with expected features
        X = pd.DataFrame(columns=all_features) 
        
        #Drop positionNamein favor for one hot encoded version
        X.drop(columns=['positionName','outsPitching','baseOnBallsPitching','hitsPitching','runsPitching',
                       'homeRunsPitching','inningsPitched'],inplace=True)     
        
        X[expected_positions] = None    
        X[['noHitter','pitchingGameScore','inningsPitchedAsFrac','gamePk2','numGames']] = None
    else:
        #Extract features of interest from dataframe
        X = df.loc[:,all_features]

        # Set dtypes
        X = X.astype({name: np.float32 for name in float_features})
        X = X.astype({name: str for name in cat_features})

        #Strip and lower case strings
        for name in cat_features:
            X.loc[:,name] = X.loc[:,name].str.strip().str.lower()

        #bin teams to major league and non-major league (if any)
        X['teamId'] = X['teamId'].apply(lambda x:bin_teams(x,team_ids))

        #Filter to players that are in the player ids of interest
        X = X[X['playerId'].isin(player_ids)]
        
        #Ensure postion names are as expected
        X = X[X['positionName'].isin(positions)]

        # Add in field for innings pitched as fraction (better for aggregation)
        X['inningsPitchedAsFrac'] = np.where(
          pd.isna(X['inningsPitched']),
          np.nan,
          np.floor(X['inningsPitched']) +
            (X['inningsPitched'] -
              np.floor(X['inningsPitched'])) * 10/3
          )
        
        X['pitchingGameScore'] = np.where(
          # pitching game score doesn't apply if player didn't pitch, set to NA
          pd.isna(X['pitchesThrown']) | 
            (X['pitchesThrown'] == 0),
          np.nan,
          (40
            + 2 * X['outsPitching']
            + 1 * X['strikeOutsPitching']
            - 2 * X['baseOnBallsPitching']
            - 2 * X['hitsPitching']
            - 3 * X['runsPitching']
            - 6 * X['homeRunsPitching']
            )
          )

        # Add in criteria for no-hitter by pitcher (individual, not multiple pitchers)
        X['noHitter'] = np.where(
          (X['completeGamesPitching'] == 1) &
          (X['inningsPitched'] >= 9) &
          (X['hitsPitching'] == 0),
          1, 0
          )      
        
        #agregate multiple games per day
        X = pd.merge(
        (X.
        groupby(['dailyDataDate', 'playerId'], as_index = False).
        # Some aggregations that are not simple sums
        agg(
          numGames = ('gamePk', 'nunique'),
          # Should be 1 team per player per day, but adding here for 1 exception:
          # playerId 518617 (Jake Diekman) had 2 games for different teams marked
          # as played on 5/19/19, due to resumption of game after he was traded
          #numTeams = ('gameTeamId', 'nunique'),
          # Should be only 1 team for all player-dates, taking min to make sure
          teamId = ('teamId', 'min'),
          positionName = ('positionName',agg_positions),
          gamePk = ('gamePk',gamePk),
          gamePk2 = ('gamePk',gamePk2)
          )
        ),
        # Merge with a bunch of player stats that can be summed at date/player level
        (X.
        groupby(['dailyDataDate', 'playerId'], as_index = False)
        [float_features + ['noHitter','pitchingGameScore','inningsPitchedAsFrac']].
        sum()
        ),
        on = ['dailyDataDate', 'playerId'],
        how = 'inner'
        )
        
        X.drop(columns=['outsPitching','baseOnBallsPitching','hitsPitching','runsPitching',
                       'homeRunsPitching','inningsPitched'],inplace=True)    

        
        X['gamePk2'] = X['gamePk2'].astype(str).str.strip().str.lower()
        
        
        #Rename positionName column to be more specific
        X.rename(columns={'positionName':'positionName_playedInGame'},inplace=True)

        #Perform NaN processing on specific features, fill with 0
        #Players either played in the game or not, so if NaN, assume they did not play
        features = ['gamesPlayedBatting',
                    'gamesPlayedPitching']
        for i,feature in enumerate(features):
                X[feature].fillna(0,inplace=True)

        #Perform NaN processing on remaining features
        #If Nan, indicates player did not have opportunity to obtain stat (e.g. a DH getting a save)
        #so encode with -1
        remaining_features = ['shutoutsPitching',
                              'gamesStartedPitching',
                              'completeGamesPitching',
                              'winsPitching',
                              'saveOpportunities',
                              'saves',
                              'blownSaves',
                              'errors', 
                              'chances',
                              'putOuts',
                              'assists',
                              'caughtStealingPitching',
                              'atBats', 
                              'plateAppearances',
                              'hits',
                              'homeRuns',
                              'doubles',
                              'triples',
                              'baseOnBalls',
                              'strikeOuts',
                              'intentionalWalks',
                              'hitByPitch',
                              'rbi',
                              'flyOuts',
                              'leftOnBase',
                              'sacBunts',
                              'sacFlies',
                              'groundOuts',
                              'stolenBases',
                              'runsScored',
                              'totalBases',
                              'caughtStealing', 
                              'pickoffs',
                              'inningsPitchedAsFrac',
                              'strikeOutsPitching',
                              'battersFaced',
                              'strikes',
                              'balls',
                              'pitchesThrown',
                              'wildPitches',
                              'earnedRuns',
                              'noHitter',
                              'pitchingGameScore'] 
        for i,feature in enumerate(remaining_features):
            X[feature] = X[feature].fillna(-1)

        #One Hot encode position played
        X = pd.get_dummies(X,columns=['positionName_playedInGame'])

        #If instance of expected features are not present, add in.
        for position in expected_positions:
            if position not in X.columns:
                X[position] = 0
            
    return X

#=======================================================================================
#Merge X_master and X_playerBoxScores, and process NaNs
def merge_master_playerBoxScores_process_NaNs(X_master, X_playerBoxScores, training_master=None ):
    #Check if X_playerBoxScores is empty. No need to check X_playerInfo since this will always exist.
    if X_playerBoxScores.empty:      
        #Add in additional columns that would've been added by merge, and fill in with -1 (does not make sense to bfill or ffill)       
        X_playerBoxScores_cols = list(X_playerBoxScores.columns)
        X_playerBoxScores_cols.remove('dailyDataDate')
        X_playerBoxScores_cols.remove('playerId')
        X_playerBoxScores_cols.remove('teamId')
        X_master[X_playerBoxScores_cols] = -1
    else:
        X_master = X_master.merge(X_playerBoxScores,how='outer',left_on = ['dailyDataDate','playerId','teamId'],
                                  right_on = ['dailyDataDate','playerId','teamId']).reset_index(drop=True)
        #Note that size of X_master increases due to double headers on certain days and other reason
        #explained below:
        #There are a small number of dailyDataDate / playerId instances where they are incorrectly duplicates due to different teamIDs
        # This is because the boxScores df has this playerID on an unknown team (non major league),
        #and the roster dataset has another team
        #Some of these are due to the all star game: roster has teamID as their normal team they play for, and box scores has it as the american league
        # (or national league) all star team
        #if training_master is None:
        allstar_game_dates = [datetime.datetime(2017,7,11),datetime.datetime(2018,7,17),
                              datetime.datetime(2019,7,9),datetime.datetime(2021,7,13)]
        X_master.reset_index(drop=True,inplace=True)
        duplicate_dates_bools = X_master[['dailyDataDate','playerId']].duplicated(keep=False)
        X_master_dup_subset = X_master[duplicate_dates_bools]
        inds_asg_dups = X_master_dup_subset[X_master_dup_subset['dailyDataDate'].isin(allstar_game_dates)].index
        asg_dups = X_master.loc[inds_asg_dups,:]
        inds_unknown = asg_dups[asg_dups['teamId'] == 'other'].index
        X_master.loc[inds_unknown,'teamId'] = np.nan
        X_master.loc[inds_asg_dups,:] = X_master.loc[inds_asg_dups,:] \
                                                .groupby(['playerId','dailyDataDate']) \
                                                .apply(lambda x: x.fillna(method='ffill')
                                                .fillna(method='bfill'))
        X_master.drop(index=inds_unknown,inplace=True)
        X_master.reset_index(drop=True,inplace=True)
        print(X_master.shape)

        #Other duplicates are due there being a difference in the teamID
        # for that specific date in the X_master and playerBoxScores.
        #X_master would've filled nan by filling forward, but actually couldve been traded and been with a new team,
        #just never appeared on roster until that date with conflict
        duplicate_dates_bools = X_master[['dailyDataDate','playerId']].duplicated(keep=False)
        X_master_dup_subset = X_master[duplicate_dates_bools]    
        inds_dups = X_master_dup_subset[(X_master_dup_subset['gamePk'].isna()) | 
                                        (X_master_dup_subset['gamePk2'].isna()) |
                                        (X_master_dup_subset['weight'].isna())].index
        X_master.loc[inds_dups,:] = X_master.loc[inds_dups,:] \
                                                 .groupby(['playerId','dailyDataDate']) \
                                                 .apply(lambda x: x.fillna(method='ffill')
                                                 .fillna(method='bfill'))        
        inds_drop = X_master.loc[inds_dups,:].groupby(['playerId','dailyDataDate']) \
                                             .apply(lambda x:min(x.index)).reset_index()
        if not inds_drop.empty:
            inds_drop = inds_drop.iloc[:,2].values
            X_master.drop(index=inds_drop,inplace=True)
            X_master.reset_index(drop=True,inplace=True)

        #Further NaN processing
        #add in flag for if player played in game that day (can treat this as a binary feature that "activates" stats features)
        X_master['OnGameRoster?'] = np.where(X_master['gamePk'].notna(),1,0)
        boxScores_features = list(X_playerBoxScores.columns)
        boxScores_features.remove('dailyDataDate')
        boxScores_features.remove('playerId')
        boxScores_features.remove('teamId')
        
        #Keep in original stats where game was played, if not, fill in with -1
        for feature in boxScores_features:
            X_master[feature] = X_master[feature].where(X_master['OnGameRoster?'] == 1,-1)

        #Forward fill / backfill static features (players dataset) for a given playerId
        #Sort by playerId and dailydatadate
        if training_master is not None:
            #print(training_master)
            cols = list(X_master.columns)
            X_master = pd.concat([training_master,X_master],axis=0,ignore_index=True)
            inds_orig_training = training_master.index
            X_master = X_master[cols].sort_values(by=['playerId','dailyDataDate',])
            X_master[['mlbDebutDate_DaysRelative','heightInches','weight','BornInUS?',
                      'primaryPositionCode_1','primaryPositionCode_10','primaryPositionCode_2',
                      'primaryPositionCode_3','primaryPositionCode_4','primaryPositionCode_5',
                      'primaryPositionCode_6','primaryPositionCode_7','primaryPositionCode_8',
                      'primaryPositionCode_9','primaryPositionCode_i','primaryPositionCode_o']] \
                     = X_master.groupby('playerId')[['mlbDebutDate_DaysRelative','heightInches','weight','BornInUS?',
                                 'primaryPositionCode_1','primaryPositionCode_10','primaryPositionCode_2',
                                 'primaryPositionCode_3','primaryPositionCode_4','primaryPositionCode_5',
                                 'primaryPositionCode_6','primaryPositionCode_7','primaryPositionCode_8',
                                 'primaryPositionCode_9','primaryPositionCode_i','primaryPositionCode_o']] \
                                 .fillna(method='ffill').fillna(method='bfill')
            X_master.drop(index=inds_orig_training,inplace=True)
        else:
            X_master.sort_values(by=['playerId','dailyDataDate'],inplace=True)        
            X_master[['mlbDebutDate_DaysRelative','heightInches','weight','BornInUS?',
                      'primaryPositionCode_1','primaryPositionCode_10','primaryPositionCode_2',
                      'primaryPositionCode_3','primaryPositionCode_4','primaryPositionCode_5',
                      'primaryPositionCode_6','primaryPositionCode_7','primaryPositionCode_8',
                      'primaryPositionCode_9','primaryPositionCode_i','primaryPositionCode_o']] \
                     = X_master.groupby('playerId')[['mlbDebutDate_DaysRelative','heightInches','weight','BornInUS?',
                                 'primaryPositionCode_1','primaryPositionCode_10','primaryPositionCode_2',
                                 'primaryPositionCode_3','primaryPositionCode_4','primaryPositionCode_5',
                                 'primaryPositionCode_6','primaryPositionCode_7','primaryPositionCode_8',
                                 'primaryPositionCode_9','primaryPositionCode_i','primaryPositionCode_o']] \
                                 .fillna(method='ffill').fillna(method='bfill')

        #Fill na's for rosters dataset
        #If you're on game roster, then you're active, and vice versa
        inds_OnGameRoster = X_master[X_master['OnGameRoster?'] == 1].index
        X_master.loc[inds_OnGameRoster,['statusCode_dl','statusCode_other','statusCode_rm']] \
                     = X_master.loc[inds_OnGameRoster,['statusCode_dl','statusCode_other','statusCode_rm']].fillna(0)
        X_master.loc[inds_OnGameRoster,'statusCode_a'] = X_master.loc[inds_OnGameRoster,'statusCode_a'].fillna(1)
        inds_OnGameRoster = X_master[X_master['OnGameRoster?'] == 0].index        
        X_master.loc[inds_OnGameRoster,'statusCode_a'] = X_master.loc[inds_OnGameRoster,'statusCode_a'].fillna(0)   
        X_master.drop(columns=['OnGameRoster?'],inplace=True)
        X_master.reset_index(drop=True,inplace=True)
    return X_master

#=======================================================================================
#Player twitters
def make_playerTwitters(df, date_features, float_features, bool_features,
                        cat_features, player_ids):
    all_features = date_features + float_features + bool_features + cat_features

    #Main processing
    if df.empty: #Create empty dataframe with expected features
        X = pd.DataFrame(columns=all_features)          
    else:
        #Extract features of interest from dataframe
        X = df[all_features]

        # Set dtypes
        X = X.astype({name: np.float32 for name in float_features})
        X = X.astype({name: str for name in cat_features})
        
        #Strip and lower case strings
        for name in cat_features:
            X.loc[:,name] = X.loc[:,name].str.strip().str.lower()

        #Filter to players that are in the player ids of interest
        X = X[X['playerId'].isin(player_ids)]

    #Rename number of followers
    X.rename(columns={'numberOfFollowers':'n_PTFollowers'},inplace=True)
    return X

def process_nans_PlayerTwitters(x,series_dates_twitterPoints,last_date):
    date = x.iloc[0]
    playerId = x.iloc[1]
    dates_twitterPoints = series_dates_twitterPoints.loc[playerId]
    if date < dates_twitterPoints[0]:
        return -1
    elif (date > dates_twitterPoints[-1]) & (dates_twitterPoints[-1] != last_date):
        return -1
    else:
        return np.nan
#=======================================================================================    
#Merge X_master and X_playerTwitters, and process NaNs
def merge_master_playerTwitters_process_NaNs(X_master, X_playerTwitters, training_master=None):
    #Check if X_playerTwitters is empty. 
    if X_playerTwitters.empty:    
        #Add in additional columns that would've been added by merge, and ffill with values from training_master 
        X_master['n_PTFollowers'] = np.nan
        if training_master is not None:      
            cols = list(X_master.columns)
            X_master = pd.concat([training_master,X_master],axis=0,ignore_index=True)
            inds_orig_training = training_master.index
            X_master = X_master[cols].sort_values(by=['playerId','dailyDataDate',])
            X_master['n_PTFollowers'] = X_master.groupby('playerId')['n_PTFollowers'] \
                                 .fillna(method='ffill').fillna(method='bfill')
            X_master.drop(index=inds_orig_training,inplace=True)
            #Fill remaining with -1 (if they were not present in training master)
            X_master['n_PTFollowers'].fillna(-1,inplace=True)                                     
        
    else:
        X_master = X_master.merge(X_playerTwitters,how='outer',left_on = ['dailyDataDate','playerId'],
                                                               right_on =['dailyDataDate','playerId'])
        X_master.reset_index(drop=True,inplace=True)
        #Get index of playerIds that are not in the playerTwitters dataset, and set to -1
        inds = X_master[~X_master['playerId'].isin(X_playerTwitters['playerId'].unique())].index
        X_master.loc[inds,'n_PTFollowers'] = -1
        
        #Fill twitter nans with -1 for players that have / have had twitters
        #Logic here is to linearly interpolate between days that have data points on either side.
        #Where this is not true, set to -1
    if training_master is None:
        series_dates_twitterPoints = X_master[X_master['n_PTFollowers'].notnull()] \
                                              .groupby('playerId')['dailyDataDate'] \
                                              .apply(lambda x:sorted(list(set(x))))
        last_date = series_dates_twitterPoints.iloc[0][-1]
        inds =  X_master[X_master['playerId'].isin(X_playerTwitters['playerId'].unique())]['n_PTFollowers'].isna()
        inds = inds.where(inds==True).dropna().index
        X_master.loc[inds,'n_PTFollowers'] = X_master.loc[inds,['dailyDataDate','playerId']] \
                                                     .apply(lambda x:process_nans_PlayerTwitters(x,
                                                     series_dates_twitterPoints,last_date),axis=1)
        X_master.sort_values(by=['playerId','dailyDataDate'],inplace=True)
        X_master.loc[:,'n_PTFollowers'] = X_master.groupby('playerId')['n_PTFollowers'].apply(
                                                   lambda x:x.interpolate(method='linear', 
                                                   limit_direction='forward', axis=0))
        del(inds,last_date,series_dates_twitterPoints)
    else:
        X_master = X_master.sort_values(by=['playerId','dailyDataDate',])
        X_master['n_PTFollowers'] = X_master.groupby('playerId')['n_PTFollowers'] \
                                 .fillna(method='ffill')
    X_master.reset_index(drop=True,inplace=True)
        
    return X_master
    
#=======================================================================================
#Team twitters
def make_teamTwitters(df, date_features, float_features, bool_features, cat_features, team_ids):
    all_features = date_features + float_features + bool_features + cat_features

    #Main processing
    if df.empty: #Create empty dataframe with expected features
        print('teamTwitters df is null')
        X = pd.DataFrame(columns=all_features)     
    else:
        print('teamTwitters df is not null')
        #Extract features of interest from dataframe
        X = df[date_features + float_features + bool_features + cat_features]
        
        # Set dtypes
        X = X.astype({name: np.float32 for name in float_features})
        X = X.astype({name: str for name in cat_features})
        
        #Strip and lower case strings
        for name in cat_features:
            X.loc[:,name] = X.loc[:,name].str.strip().str.lower()
            
        #bin teams to major league and non-major league (if any)
        X['teamId'] = X['teamId'].apply(lambda x:bin_teams(x,team_ids))
        
    #Rename column
    X.rename(columns={'numberOfFollowers':'n_TTFollowers'},inplace=True)
    return X

#=======================================================================================    
#Merge X_master and X_teamTwitters, and process NaNs
def merge_master_teamTwitters_process_NaNs(X_master, X_teamTwitters, training_master=None):
    #Check if X_playerTwitters is empty. No need to check X_playerInfo since this will always exist.
    if X_teamTwitters.empty:
        #Add in additional columns that would've been added by merge, and fill in with -1 (convention for NaN)        
        X_master['n_TTFollowers'] = np.nan
        if training_master is not None:      
            cols = list(X_master.columns)
            X_master = pd.concat([training_master,X_master],axis=0,ignore_index=True)
            inds_orig_training = training_master.index
            X_master = X_master[cols].sort_values(by=['playerId','dailyDataDate',])
            X_master['n_TTFollowers'] = X_master.groupby('playerId')['n_TTFollowers'] \
                                 .fillna(method='ffill').fillna(method='bfill')
            X_master.drop(index=inds_orig_training,inplace=True)
            #Fill remaining with -1 (if they were not present in training master)
            X_master['n_TTFollowers'].fillna(-1,inplace=True)    
    else:
        X_master = X_master.merge(X_teamTwitters,how='outer',left_on = ['dailyDataDate','teamId'],
                                                             right_on = ['dailyDataDate','teamId'])

        inds = X_master[~X_master['teamId'].isin(X_teamTwitters['teamId'].unique())].index
        X_master.loc[list(inds),'n_TTFollowers'] = -1
        
    if training_master is None: 
        # linearly interpolate remaining NaNs
        X_master.sort_values(by=['playerId','dailyDataDate'],inplace=True)
        X_master.loc[:,'n_TTFollowers'] = X_master.groupby('playerId')['n_TTFollowers'].apply(
                                            lambda x:x.interpolate(method='linear', limit_direction='forward', axis=0))
    else:
        X_master = X_master.sort_values(by=['playerId','dailyDataDate',])
        X_master['n_TTFollowers'] = X_master.groupby('playerId')['n_TTFollowers'] \
                                 .fillna(method='ffill')
    X_master.reset_index(drop=True,inplace=True)
        
    return X_master

#=======================================================================================
#Awards
def make_awards(awards, awards_static, player_ids):

    #Main processing
    if awards.empty: #Create empty dataframe with expected features
        print('awards df is null')
        player_selected_awards_by_date = pd.DataFrame(
                                        columns=['dailyDataDate','playerId',    
                                                 'toDateAllStars','toDateCyYoungs','toDateMVPs','toDatePlayerOfWeeks',
                                                 'toDateWorldSeriesChampionships','toDatePlayerOfMonths',
                                                 'toDateSilverSluggers','toDateWorldSeriesMVPs','toDateLCSMVPs',
                                                 'toDateRookieOfYears','toDateMinorLeagueAccolades'])   
    else:
        #Extract features of interest from dataframe
        X = awards[['dailyDataDate','awardId','playerId','awardPlayerTeamId']]
        awards_static.rename(columns={'awardDate':'dailyDataDate'},inplace=True)
        awards_static = awards_static[['dailyDataDate','awardId','playerId','awardPlayerTeamId']]
        X = pd.concat((awards_static,X),axis=0,ignore_index=True)
        
        # Set dtypes
        X = X.astype({'playerId':str,'awardId':str,'awardPlayerTeamId':str})
        
        #Filter to tracked players
        print('Number of rows before filtering to tracked players: ',X.shape[0])
        X = X[X['playerId'].isin(player_ids)]
        print('Number of rows after filtering to tracked players: ',X.shape[0])

        selected_awards = pd.DataFrame(data = {
          'awardId':  ['ALAS', 'NLAS', 'ALMVP', 'NLMVP', 'ALCY', 'NLCY',
                       'NLPOW','ALPOW','WSCHAMP','NLPOM','ALPOM',
                       'NLPITOM','ALPITOM','NLRRELMON','ALRRELMON','NLROM','ALROM',
                       'NLSS','ALSS','WSMVP','NLCSMVP','ALCSMVP',
                       'ALROY','NLROY','BAMLROY',
                       'MILBORGAS','FUTURES','BAMILAS','AFLRS'],
          'awardCategory': ['AllStar', 'AllStar', 'MVP', 'MVP', 'CyYoung', 'CyYoung',
                            'PlayerOfWeek','PlayerOfWeek','WorldSeriesChampionship','PlayerOfMonth','PlayerOfMonth',
                            'PlayerOfMonth','PlayerOfMonth','PlayerOfMonth','PlayerOfMonth','PlayerOfMonth','PlayerOfMonth',
                            'SilverSlugger','SilverSlugger','WorldSeriesMVP','LCSMVP','LCSMVP',
                            'RookieOfYear','RookieOfYear','RookieOfYear',
                            'MinorLeagueAccolade','MinorLeagueAccolade','MinorLeagueAccolade','MinorLeagueAccolade']
          })
    
        player_selected_awards = pd.merge(
          X,
          selected_awards,
          on = 'awardId',
          # Inner join to limit player awards to only selected ones
          how = 'inner'
          )
        selected_award_categories_in_data = (player_selected_awards['awardCategory'].
          unique())

        player_selected_awards_by_date = (player_selected_awards.
          # Add count for use when pivoting
          assign(count = 1).
          pivot_table(
            index = ['dailyDataDate', 'playerId'],
            columns = 'awardCategory',
            values = 'count',
            # NA can be turned to 0 since it means player didn't get that award that day
            fill_value = 0
          ).
          reset_index()
          )

        # Add cumulative 'to date' sums for each award category
        for award_category in selected_award_categories_in_data:
            player_selected_awards_by_date[('toDate' + award_category + 's')] = (
              player_selected_awards_by_date.
                groupby(['playerId'])[award_category].cumsum()
              )

        player_selected_awards_by_date.drop(selected_award_categories_in_data,
          axis = 1, inplace = True)
        player_selected_awards_by_date = player_selected_awards_by_date[player_selected_awards_by_date['dailyDataDate']>='2018-01-01']
    return player_selected_awards_by_date

#=======================================================================================    
#Merge X_master and X_awards, and process NaNs
def merge_master_awards_process_NaNs(X_master, X_awards, training_master=None):
    #Check if X_awards is empty. 
    awards_col = list(X_awards.columns)
    awards_col.remove('dailyDataDate')
    awards_col.remove('playerId')
    if X_awards.empty:
        #Add in additional columns that would've been added by merge, and fill in with -1 (convention for NaN)      
        X_master[awards_col] = np.nan
        if training_master is not None:      
            cols = list(X_master.columns)
            X_master = pd.concat([training_master,X_master],axis=0,ignore_index=True)
            inds_orig_training = training_master.index
            X_master = X_master[cols].sort_values(by=['playerId','dailyDataDate',])
            X_master[awards_col] = X_master.groupby('playerId')[awards_col] \
                                 .fillna(method='ffill').fillna(method='bfill')
            X_master.drop(index=inds_orig_training,inplace=True)
    else:
        X_master = X_master.merge(X_awards,how='outer',left_on = ['dailyDataDate','playerId'],
                                                       right_on = ['dailyDataDate','playerId'])
        inds = X_master[~X_master['playerId'].isin(X_awards['playerId'].unique())].index
        X_master.loc[inds,awards_col] = 0
        
        # linearly interpolate remaining NaNs
        X_master.sort_values(by=['playerId','dailyDataDate'],inplace=True)
        X_master.loc[:,awards_col] = X_master.groupby('playerId')[awards_col] \
                                                   .apply(lambda x:x.fillna(method='ffill',axis=0))
        #fillna remaining Nans with 0
        X_master.fillna(0,inplace=True)
    X_master.reset_index(drop=True,inplace=True)
        
    return X_master

#=======================================================================================
#Seasons
def make_seasons(seasons,X_master):
    #No need to check for missing dataframes, as both of these will always be available
    dates = pd.DataFrame(data = {'dailyDataDate': X_master['dailyDataDate'].unique()})
    dates['year'] = dates['dailyDataDate'].dt.year
    dates['month'] = dates['dailyDataDate'].dt.month
    dates_with_info = pd.merge(
      dates,
      seasons,
      left_on = 'year',
      right_on = 'seasonId',
      how = 'inner'
      )
    # Count anything between regular and Postseason as "in season"
    dates_with_info['inSeason'] = (
      dates_with_info['dailyDataDate'].between(
        dates_with_info['regularSeasonStartDate'],
        dates_with_info['postSeasonEndDate'],
        inclusive = True
        )
      )
    dates_with_info['AllStarGame'] = dates_with_info['dailyDataDate'] == dates_with_info['allStarDate']
    # Separate dates into different parts of MLB season
    dates_with_info['seasonPart'] = np.select(
      [
        dates_with_info['dailyDataDate'] < dates_with_info['preSeasonStartDate'], 
        dates_with_info['dailyDataDate'] < dates_with_info['regularSeasonStartDate'],
        dates_with_info['dailyDataDate'] <= dates_with_info['lastDate1stHalf'],
        dates_with_info['dailyDataDate'] < dates_with_info['firstDate2ndHalf'],
        dates_with_info['dailyDataDate'] <= dates_with_info['regularSeasonEndDate'],
        dates_with_info['dailyDataDate'] < dates_with_info['postSeasonStartDate'],
        dates_with_info['dailyDataDate'] <= dates_with_info['postSeasonEndDate'],
        dates_with_info['dailyDataDate'] > dates_with_info['postSeasonEndDate']
      ], 
      [
        'Offseason',
        'Preseason',
        'Reg_Season_1st_Half',
        'All-Star_Break',
        'Reg_Season_2nd_Half',
        'Between_Reg_and_Postseason',
        'Postseason',
        'Offseason'
      ], 
      default = np.nan
      )
    dates_with_season_part = (dates_with_info[['dailyDataDate', 'year',
      'seasonId', 'month', 'inSeason', 'seasonPart','AllStarGame']].
      rename(columns = {'seasonId': 'season'})
      )
    dates_with_season_part = pd.get_dummies(dates_with_season_part,columns=['seasonPart'])
    #Add in seasons not present
    expected_seasons = ['seasonPart_All-Star_Break','seasonPart_Between_Reg_and_Postseason',
                        'seasonPart_Offseason','seasonPart_Postseason','seasonPart_Preseason',
                        'seasonPart_Reg_Season_1st_Half','seasonPart_Reg_Season_2nd_Half']
    for season in expected_seasons:
        if season not in dates_with_season_part.columns:
            dates_with_season_part[season] = 0
    dates_with_season_part.drop(columns='season',inplace=True)
    return dates_with_season_part

#=======================================================================================
#Team box score
def make_teamBoxScores(df, date_features, float_features, bool_features,
                       cat_features, team_ids):
    all_features = date_features + float_features + bool_features + cat_features

    #Main processing
    if df.empty: #Create empty dataframe with expected features
        all_features.append('gameTime')
        X = pd.DataFrame(columns=all_features + ['gamePk2'])     
    else:
        print('teamBoxScores df is not null')
        X = df.loc[:, all_features]
        #Extract gametime
        X['gameTimeUTC'] = (pd.to_datetime(X['gameTimeUTC'],format='%Y-%m-%d',utc=True).dt.hour)*60 \
                            + pd.to_datetime(X['gameTimeUTC'],format='%Y-%m-%d',utc=True).dt.minute
        
        # Set dtypes
        X = X.astype({name: np.float32 for name in float_features})
        X = X.astype({name: str for name in cat_features})        
        
        #Strip and lower case strings
        for name in cat_features:
            X.loc[:,name] = X.loc[:,name].str.strip().str.lower()
            
        X = pd.merge(
        (X.
        groupby(['dailyDataDate', 'teamId',], as_index = False).
        # Some aggregations that are not simple sums
        agg(
          #numGames = ('gamePk', 'nunique'),
          # Should be 1 team per player per day, but adding here for 1 exception:
          # playerId 518617 (Jake Diekman) had 2 games for different teams marked
          # as played on 5/19/19, due to resumption of game after he was traded
          #numTeams = ('gameTeamId', 'nunique'),
          # Should be only 1 team for all player-dates, taking min to make sure
          teamId = ('teamId', 'min'),
          gameTimeUTC = ('gameTimeUTC',gamePk),
          gamePk = ('gamePk',gamePk),
          gamePk2 = ('gamePk',gamePk2)
          )
        ),
        # Merge with a bunch of player stats that can be summed at date/player level
        (X.
        groupby(['dailyDataDate', 'teamId'], as_index = False)
        [float_features].
        sum()
        ),
        on = ['dailyDataDate', 'teamId'],
        how = 'inner'
        )
        
        X['home'] = X['home'].clip(0,1)
        X['gamePk2'] = X['gamePk2'].astype(str).str.strip().str.lower()

        
        #bin teams to major league and non-major league (if any)
        X['teamId'] = X['teamId'].apply(lambda x:bin_teams(x,team_ids))    

    #rename columns
    X.rename(columns={'hits':'TeamHits',
                      'doubles':'TeamDoubles',
                      'triples':'TeamTriples',
                      'homeRuns':'TeamHomeRuns',
                      'strikeOuts':'TeamStrikeOuts',
                      'baseOnBalls':'TeamBaseOnBalls',
                      'intentionalWalks':'TeamIntentionalWalks',
                      'hitByPitch':'TeamHBP',
                      'runsScored':'TeamRunsScored',
                      'stolenBases':'TeamStolenBases',
                      'leftOnBase':'TeamLeftOnBase',
                      'strikeOutsPitching':'TeamStrikeOutsPitching',
                      'pickoffs':'TeamPickoffs',
                      'home':'HomeTeam'},inplace=True)
    return X

#=======================================================================================    
#Merge X_master and X_teamBoxScores, and process NaNs
def merge_master_teamBoxScores_process_NaNs(X_master, X_teamBoxScores,training_master=None):
    #Check if X_teamBoxScores is empty. 
    if X_teamBoxScores.empty:
        #Add in additional columns that would've been added by merge, and fill in with -1
        X_teamBoxScores_cols = list(X_teamBoxScores.columns)
        X_teamBoxScores_cols.remove('dailyDataDate')
        X_teamBoxScores_cols.remove('gamePk')
        X_teamBoxScores_cols.remove('gamePk2')
        X_teamBoxScores_cols.remove('teamId')
        X_master[X_teamBoxScores_cols] = -1
    else:
        X_master = X_master.merge(X_teamBoxScores,how='left',left_on = ['dailyDataDate','gamePk','gamePk2','teamId'],
                                                             right_on = ['dailyDataDate','gamePk','gamePk2','teamId'])   
        X_master.fillna(-1,inplace=True)
        X_master.isna().any()[X_master.isna().any() == True]
        X_master.reset_index(drop=True,inplace=True)

    return X_master

#=======================================================================================
def gamePk(arg):
    if arg.nunique() == 1:
        return arg.unique()[0]
    else:
        return min(arg)

    
def gamePk2(arg):
    if arg.nunique() == 1:
        return -1
    else:
        return max(arg)
    
def agg_gameStats(arg):
    if arg.nunique() == 1:
        return arg.unique()[0]
    elif arg.nunique() >= 2:
        return arg.iloc[-1]

#Game info
def make_gameInfo(df, date_features, float_features, bool_features, 
                  cat_features, team_ids):

    all_features = date_features + float_features + bool_features + cat_features

    #Main processing
    if df.empty: #Create empty dataframe with expected features
        X = pd.DataFrame(columns=all_features+['gamePk2'])    
    else:
        X = df.loc[:,all_features + ['codedGameState','gameTimeUTC']]
        X['gameTimeUTC'] = (pd.to_datetime(X['gameTimeUTC'],format='%Y-%m-%d',utc=True).dt.hour)*60 
        X['gameTimeUTC'] += pd.to_datetime(X['gameTimeUTC'],format='%Y-%m-%d',utc=True).dt.minute
        #Remove games with gamestates of D C S or U, corresponding to Postponed, Scheduled, Cancelled, or Suspended games
        X = X[X['homeScore'].notna()]
        X = X[X['codedGameState'] == 'F']
        X.drop(columns=['codedGameState'],inplace=True)

        
        #Esnure winner values are as expected, if not set to 'N'
        inds = X[~X['awayWinner'].isin([True,False])].index
        X.loc[inds,'awayWinner'] = False
        inds = X[~X['homeWinner'].isin([True,False])].index
        X.loc[inds,'homeWinner'] = False
        
        #Drop duplicates
        inds = X[['dailyDataDate','gamePk']].duplicated()
        X = X[~inds]
        
        # Set dtypes
        X = X.astype({name: np.float32 for name in float_features})
        X = X.astype({name: str for name in cat_features})
        X = X.astype({name: bool for name in bool_features})
        
        #Strip and lower case strings
        for name in cat_features:
            X.loc[:,name] = X.loc[:,name].str.strip().str.lower()

        
        X.sort_values(by=['dailyDataDate','gameTimeUTC'])
        X = (X.groupby(['dailyDataDate', 'awayId'], as_index = False). 
        # Some aggregations that are not simple sums
        agg(
          #numGames = ('gamePk', 'nunique'),
          # Should be 1 team per player per day, but adding here for 1 exception:
          # playerId 518617 (Jake Diekman) had 2 games for different teams marked
          # as played on 5/19/19, due to resumption of game after he was traded
          #numTeams = ('gameTeamId', 'nunique'),
          # Should be only 1 team for all player-dates, taking min to make sure
          homeId = ('homeId','min'),
          gamePk = ('gamePk',gamePk),
          gamePk2 = ('gamePk',gamePk2),
          homeWinPct = ('homeWinPct',agg_gameStats),
          awayWinPct =  ('awayWinPct',agg_gameStats),
          homeWins = ('homeWins',agg_gameStats),
          homeLosses = ('homeLosses',agg_gameStats),
          homeScore = ('homeScore',agg_gameStats),
          awayWins =  ('awayWins',agg_gameStats),
          awayLosses = ('awayLosses',agg_gameStats),
          awayScore = ('awayScore',agg_gameStats),
          homeWinner = ('homeWinner',agg_gameStats),
          awayWinner = ('awayWinner',agg_gameStats),
          ))
        
        X['gamePk2'] = X['gamePk2'].astype(str).str.strip().str.lower()
    
        
        #bin teams to major league and non-major league
        X['awayId'] = X['awayId'].apply(lambda x:bin_teams(x,team_ids))
        X['homeId'] = X['homeId'].apply(lambda x:bin_teams(x,team_ids))
                             
    return X
    
#=======================================================================================    
#Merge X_master and X_gameInfo, and process NaNs, Additional feature engineering
def merge_master_gameInfo_process_NaNs(X_master, X_gameInfo, training_master=None):
    #Check if X_teamBoxScores is empty. 
    if X_gameInfo.empty:
        #Add in additional columns that would've been added by merge, and fill in with -1 (convention for NaN)  
        addtl_cols = list(X_gameInfo.columns)
        addtl_cols.remove('dailyDataDate')
        addtl_cols.remove('gamePk')
        addtl_cols.remove('gamePk2')
        X_master[addtl_cols] = -1  
    else:
        X_master = X_master.merge(X_gameInfo, how='left', left_on = ['dailyDataDate','gamePk','gamePk2'],
                                                             right_on = ['dailyDataDate','gamePk','gamePk2']) 
        X_master.fillna(-1,inplace=True)
    X_master.reset_index(drop=True,inplace=True)
        
    return X_master
  
def encode_opponent(X_master,newColumn,homeColumn,awayColumn):
    inds = X_master[X_master['HomeTeam'] > 0.5]['HomeTeam'].index #Return awayId
    inds2 = X_master[(X_master['HomeTeam'] <= 0.5) & (X_master['HomeTeam'] >= -0.5)]['HomeTeam'].index #Return homeId
    inds3 = X_master[X_master[['HomeTeam',homeColumn,awayColumn]].isna().any(axis=1)].index
    X_master[newColumn] = np.nan
    X_master.loc[inds,newColumn] = X_master.loc[inds,awayColumn].values 
    X_master.loc[inds2,newColumn] = X_master.loc[inds2,homeColumn].values 
    X_master.loc[inds3,[newColumn,homeColumn,awayColumn]] = -1
    return X_master

def encode_team(X_master,newColumn,homeColumn,awayColumn):
    inds = X_master[X_master['HomeTeam'] > 0.5]['HomeTeam'].index #Return homeId
    inds2 = X_master[(X_master['HomeTeam'] <= 0.5) & (X_master['HomeTeam'] >= -0.5)]['HomeTeam'].index #Return awayId
    inds3 = X_master[X_master[['HomeTeam','homeWinPct','awayWinPct']].isna().any(axis=1)].index
    X_master[newColumn] = np.nan

    X_master.loc[inds,newColumn] = X_master.loc[inds,homeColumn].values 
    X_master.loc[inds2,newColumn] = X_master.loc[inds2,awayColumn].values 
    X_master.loc[inds3,[newColumn,homeColumn,awayColumn]] = -1
    return X_master

#=======================================================================================
#Standings   
def make_standings_info(df, cat_features, float_features, date_features, 
                        bool_features, team_ids):
    #Input checks
    all_features = date_features + float_features + bool_features + cat_features

    #Main processing
    if df.empty: #Create empty dataframe with expected features
        print('standingsInfo df is null')
        X = pd.DataFrame(columns=all_features)  
        X.drop(columns=['streakCode'],inplace=True)
        X['streakType'] = None
        X['streakValue'] = None
    else:
        #Extracct features
        X = df.loc[:,all_features]
        
        #Do some processing on streak data
        X['streakType'] = X['streakCode'].str.extract('(\D+)')
        X['streakValue'] = X['streakCode'].str.extract('(\d+)')
        X.replace({'-':np.nan},inplace=True)
        X.replace({'E':-1},inplace=True)
        X.drop(columns=['streakCode'],inplace=True)
        X['streakType'] = X['streakType'].fillna(-1)
        
        # Set dtypes
        float_features.append('streakValue')
        _ = cat_features.pop(0)
        cat_features.append('streakType')
        X = X.astype({name: np.float32 for name in float_features})
        X = X.astype({name: str for name in cat_features})
        X = X.astype({name: bool for name in bool_features})
        
        #Strip and lower case strings
        for name in cat_features:
            X.loc[:,name] = X.loc[:,name].str.strip().str.lower()
        
        #Retain only teams that are already in the master list. Don't need to track other teams, so just dorp those rows
        X = X[X['teamId'].isin(team_ids)]
        
        #Take care of NaNs in standings
        #Fill in standings NANs with -1 (instances where a particular stat does not apply )
        X.fillna(-1,inplace=True)
        
    return X

#=======================================================================================    
#Merge X_master and X_standings, and process NaNs
def merge_master_standings_process_NaNs(X_master, X_standings, training_master=None):
    #Check if X_teamBoxScores is empty. 
    addtl_cols = list(X_standings.columns)
    addtl_cols.remove('dailyDataDate')
    addtl_cols.remove('teamId')
    if X_standings.empty:  
        #Add in additional columns that would've been added by merge, and fill in with -1 (convention for NaN)  
        #Add in logic to fill in from previous day if in season, otherwise set to -1 (not applicable)
        X_master[addtl_cols] = np.nan
   
    else:
        X_master = X_master.merge(X_standings, how='outer', left_on = ['dailyDataDate','teamId'],
                                                            right_on = ['dailyDataDate','teamId'])
    if training_master is not None:      
        #Only fill in when we are in season, otherwise set to -1
        if X_master['inSeason'].values[0] == True:
            cols = list(X_master.columns)
            X_master = pd.concat([training_master,X_master],axis=0,ignore_index=True)
            inds_orig_training = training_master.index
            X_master = X_master[cols].sort_values(by=['playerId','dailyDataDate',])
            X_master.loc[:,addtl_cols] = X_master.groupby('playerId')[addtl_cols] \
                                     .fillna(method='ffill').fillna(method='bfill')
        X_master.drop(index=inds_orig_training,inplace=True)
    else:
        #FFill Nas for indices that are in season
        X_master.sort_values(by=['dailyDataDate','playerId'],inplace=True)
        inds = X_master[X_master['inSeason'] == True].index
        X_master.loc[inds,addtl_cols] = (X_master.loc[inds,:].groupby(['playerId','year']))[addtl_cols].fillna(method='ffill')
    X_master.fillna(-1,inplace=True)
    X_master.reset_index(drop=True,inplace=True)
        
    return X_master
      
#=======================================================================================
#Events                          

def make_events_info(events, games, cat_features, float_features, 
                     bool_features, player_ids):
    
    all_features = float_features + bool_features + cat_features
    
    #Main processing
    if events.empty: #Create empty dataframe with expected features
        X = pd.DataFrame(columns=['dailyDataDate','playerId','gamePk','gamePk2'
                                            'pitches100mph','HRDist450ft','gameTyingRBI','goAheadRBI','walkoffRBI'])  
    else:
        # Merge games w/ events to get scheduled length of game (helps w/ some calculations)
        X = pd.merge(
          events,
          games[['gamePk', 'scheduledInnings']].drop_duplicates(),
          on = ['gamePk'],
          how = 'left'
          )

        # Get current score from batting & pitching team perspectives
        X['battingTeamScore'] = np.where(X['halfInning'] == 'bottom',
          X['homeScore'], X['awayScore'])

        X['pitchingTeamScore'] = np.where(X['halfInning'] == 'bottom',
          X['awayScore'], X['homeScore'])

        X['pitches100mph'] = np.where(
          (X['type'] == 'pitch') & (X['startSpeed'] >= 100), 
          1, 0)
        X['HRDist450ft'] = np.where(
          (X['event'] == 'Home Run') & (X['totalDistance'] >= 450), 
          1, 0)

        # Use game context/score logic to add fields for notable in-game events
        X['gameTyingRBI'] = np.where(
          (X['isPaOver'] == 1) & (X['rbi'] > 0) &
          # Start w/ batting team behind in score...
          (X['battingTeamScore'] < X['pitchingTeamScore']) & 
          # ...and look at cases where adding RBI ties score
          ((X['battingTeamScore'] + X['rbi']) == 
            X['pitchingTeamScore']
            ),
          1, 0)

        X['goAheadRBI'] = np.where(
          (X['isPaOver'] == 1) & (X['rbi'] > 0) &
          # Start w/ batting team not ahead in score (can be tied)...
          (X['battingTeamScore'] <= X['pitchingTeamScore']) &
          # ... and look at cases where adding RBI puts batting team ahead
          ((X['battingTeamScore'] + X['rbi']) >
            X['pitchingTeamScore']
            ),
          1, 0)

        # Add field to count walk-off (game-winning, game-ending) RBI
        X['walkoffRBI'] = np.where(
          (X['inning'] >= X['scheduledInnings']) & 
          (X['halfInning'] == 'bottom') &
          (X['goAheadRBI'] == 1),
          1, 0)
        
        X['gameTimeUTC'] = (pd.to_datetime(X['gameTimeUTC'],format='%Y-%m-%d',utc=True).dt.hour)*60 
        X['gameTimeUTC'] += pd.to_datetime(X['gameTimeUTC'],format='%Y-%m-%d',utc=True).dt.minute

        added_events_fields = ['pitches100mph', 'HRDist450ft', 'gameTyingRBI',
          'goAheadRBI', 'walkoffRBI']

        added_events_fields = ['HRDist450ft', 'gameTyingRBI',
                  'goAheadRBI', 'walkoffRBI']
        X['gameTimeUTC'] = (pd.to_datetime(X['gameTimeUTC'],format='%Y-%m-%d',utc=True).dt.hour)*60 
        X['gameTimeUTC'] += pd.to_datetime(X['gameTimeUTC'],format='%Y-%m-%d',utc=True).dt.minute
        X1 = X[['dailyDataDate','hitterId','gamePk','gameTimeUTC'] + ['HRDist450ft', 'gameTyingRBI',
                  'goAheadRBI', 'walkoffRBI']].rename(columns={'hitterId':'playerId'})
        X1['pitches100mph'] = 0
        X1 = X1[(X1['HRDist450ft']!=0) | (X1['gameTyingRBI']!=0)
               | (X1['goAheadRBI']!=0) | (X1['walkoffRBI']!=0)]
        X2 = X[['dailyDataDate','pitcherId','gamePk','gameTimeUTC'] + ['pitches100mph']].rename(columns={'pitcherId':'playerId'})
        X2[added_events_fields] = 0
        X2 = X2[X2['pitches100mph']!=0]
        X = pd.concat((X1,X2),axis=0,ignore_index=True).sort_values(by=['dailyDataDate','gamePk','gameTimeUTC'])
        # Set dtypes
        X = X.astype({name: str for name in cat_features})
        
        #Strip and lower case strings
        for name in cat_features:
            X.loc[:,name] = X.loc[:,name].str.strip().str.lower()
            
        #Aggregate multiple instances
        X = pd.merge(
            (X.groupby(['dailyDataDate', 'playerId'], as_index = False). 
        # Some aggregations that are not simple sums
        agg(
          #numGames = ('gamePk', 'nunique'),
          # Should be 1 team per player per day, but adding here for 1 exception:
          # playerId 518617 (Jake Diekman) had 2 games for different teams marked
          # as played on 5/19/19, due to resumption of game after he was traded
          #numTeams = ('gameTeamId', 'nunique'),
          # Should be only 1 team for all player-dates, taking min to make sure
          gamePk = ('gamePk',gamePk),
          gamePk2 = ('gamePk',gamePk2),
          )),
            (X.groupby(['dailyDataDate', 'playerId'], as_index = False)
            [['pitches100mph',
            'HRDist450ft',
            'gameTyingRBI',
            'goAheadRBI',
            'walkoffRBI']].sum()
            ),
          on = ['dailyDataDate', 'playerId'],
          how = 'inner'
          )    
        
        X['gamePk2'] = X['gamePk2'].astype(str).str.strip().str.lower()
        
        # Filter to players of interest
        X = X[X['playerId'].isin(player_ids)]
        
    return X   

#=======================================================================================    
#Merge X_master and X_standings, and process NaNs
def merge_master_events_process_NaNs(X_master, X_events):#, training_master=False):
    #Check if X_teamBoxScores is empty. 
    if X_events.empty:
        #Add in additional columns that would've been added by merge, and fill in with -1 (convention for NaN)  
        cols = ['pitches100mph', 'HRDist450ft', 'gameTyingRBI',
          'goAheadRBI', 'walkoffRBI']
        X_master[cols] = -1   
    else:
        X_master = X_master.merge(X_events, how='left', left_on = ['dailyDataDate','gamePk','gamePk2','playerId'],
                                                        right_on = ['dailyDataDate','gamePk','gamePk2','playerId'])
        X_master.fillna(-1,inplace=True)
    
    X_master.reset_index(drop=True,inplace=True)        
    return X_master
#=======================================================================================    
# Transactions
def make_transactions_info(transactions, player_ids, team_ids):
        
    #Main processing
    if transactions.empty: #Create empty dataframe with expected features
        X = pd.DataFrame(columns=['dailyDataDate','playerId','assigned','claimedOffWaivers','declaredFreeAgency',
                                  'designatedforAssignment','optioned','recalled','released','selected','signed',
                                  'signedasFreeAgent','statusChange','trade','toTeamId'])  
    else:
        # Pick certain transaction codes of interest from above list
        transactions['typeCode'] = transactions['typeCode'].astype(str).str.strip().str.lower()
        transaction_codes_of_interest = ['asg','sfa','sc','opt','cu','sgn','se','tr','des','dfa','clw','rel']

        player_date_transactions_wide = (transactions.
          assign(
            # Create field w/ initial lower case & w/o spaces for later field names
            typeDescNoSpace = [(typeDesc[0].lower() + typeDesc[1:]) for typeDesc in
              transactions['typeDesc'].str.replace(' ', '')],
            # Add count ahead of pivot
            count = 1
            )
          [
          # Filter to transactions of desired types and rows for actual players
            np.isin(transactions['typeCode'], transaction_codes_of_interest) &
            pd.notna(transactions['playerId'])
          ][['dailyDataDate', 'playerId', 'typeDescNoSpace', 'count']].
          # Filter to unique transaction types across player-date
          drop_duplicates().
          # Pivot data to 1 row per player-date and 1 column per transaction type
          pivot_table(
            index = ['dailyDataDate', 'playerId'],#,'fromTeamId','toTeamId'],
            columns = 'typeDescNoSpace',
            values = 'count',
            # NA can be turned to 0 since it means player didn't have that transaction that day
            fill_value = 0
            ).
          reset_index()
          )
        player_date_transactions_wide['playerId'] = player_date_transactions_wide['playerId'].astype(float)
        player_date_transactions_wide['playerId'] = player_date_transactions_wide['playerId'].astype(int)
        player_date_transactions_wide['playerId'] = player_date_transactions_wide['playerId'].astype(str).str.strip().str.lower()
        
        teamInfo = transactions[['dailyDataDate','playerId','toTeamId']]
        inds = teamInfo[teamInfo['playerId'] != 'nan'].index
        teamInfo = teamInfo.loc[inds,:]
        teamInfo = teamInfo[teamInfo['playerId'].notnull()]
        teamInfo['playerId'] = teamInfo['playerId'].astype(float)
        teamInfo['playerId'] = teamInfo['playerId'].astype(int)
        teamInfo['playerId'] = teamInfo['playerId'].astype(str).str.strip().str.lower()
        
        teamInfo['toTeamId'].fillna(-1,inplace=True)
        
        teamInfo['toTeamId'] = teamInfo['toTeamId'].astype(float)
        teamInfo['toTeamId'] = teamInfo['toTeamId'].astype(int)
        teamInfo['toTeamId'] = teamInfo['toTeamId'].astype(str).str.strip().str.lower()
    
        
        X = player_date_transactions_wide.merge(teamInfo,how='left',left_on = ['dailyDataDate','playerId'],
                                        right_on = ['dailyDataDate','playerId'])
        dup_inds = X[X[['dailyDataDate','playerId']].duplicated(keep='last')].index
        X.drop(dup_inds,inplace=True)
        
        X['toTeamId'] = X['toTeamId'].apply(lambda x:bin_teams(x,team_ids))
        #Fill in any NaNs
        X.fillna(0,inplace=True)
        #Check for presence of all required columns:
        cols = ['assigned','claimedOffWaivers','declaredFreeAgency',
                                  'designatedforAssignment','optioned','recalled','released','selected','signed',
                                  'signedasFreeAgent','statusChange','trade']
        for col in cols:
            if col not in list(X.columns):
                X[col] = 0

    return X
#=======================================================================================    
#Merge X_master and transactions, and process NaNs        
def merge_master_transactions_process_NaNs(X_master, X_transactions):#, training_master=False):
    #Check if X_transactions is empty. 
    if X_transactions.empty:   
        #Add in additional columns that would've been added by merge, and fill in with -1 (convention for NaN)  
        cols = list(X_transactions.columns)
        cols.remove('dailyDataDate')
        cols.remove('playerId')
        X_master[cols] = 0   
    else:
        X_master = X_master.merge(X_transactions, how='left', left_on = ['dailyDataDate','playerId'],
                                                        right_on = ['dailyDataDate','playerId'])
        X_master.fillna(0,inplace=True)
    
    X_master.reset_index(drop=True,inplace=True)        
    return X_master

#=======================================================================================    
# Final processing on X_master
#Helper function to implement clustering on teams by payroll and attendance
def team_clustering(X_master,team_clusters):
    team_clusters['cluster_payroll'] = team_clusters['cluster_payroll'].astype(int)
    team_clusters['cluster_attend'] = team_clusters['cluster_attend'].astype(int)
    payroll_map = pd.Series(team_clusters.cluster_payroll.values,
                            index=team_clusters.teamId.astype(str)).to_dict()
    payroll_map['-1'] = int(-1)
    payroll_map[-1] = int(-1)
    payroll_map['other'] = int(-1)
    
    attend_map = pd.Series(team_clusters.cluster_attend.values,
                            index=team_clusters.teamId.astype(str)).to_dict()
    attend_map['-1'] = int(-1)
    attend_map[-1] = int(-1)
    attend_map['other'] = int(-1)
    
    X_master['teamId_payrollCluster'] = X_master['teamId'].map(payroll_map)
    X_master['teamId_attendCluster'] = X_master['teamId'].map(attend_map)
    X_master['OpponentId_payrollCluster'] = X_master['OpponentId'].map(payroll_map)
    X_master['OpponentId_attendCluster'] = X_master['OpponentId'].map(attend_map)
    X_master['toTeamId_payrollCluster'] = X_master['toTeamId'].map(payroll_map)
    X_master['toTeamId_attendCluster'] = X_master['toTeamId'].map(attend_map)
    X_master.drop(columns=['OpponentId','toTeamId'],inplace=True)
    return X_master
    
def final_processing(X_master, team_clusters, nextDayPlayerEngagement = None):
    
    #Convert several boolean columns to 1 or 0
    X_master['inSeason'] = X_master['inSeason'].astype(int)
    X_master['AllStarGame'] = X_master['AllStarGame'].astype(int)
    X_master['divisionChamp'] = X_master['divisionChamp'].astype(int)
    X_master['divisionLeader'] = X_master['divisionLeader'].astype(int)
    X_master['wildCardLeader'] = X_master['wildCardLeader'].astype(int)
    X_master['TeamWinner'] = X_master['TeamWinner'].astype(int)
    X_master['OpponentWinner'] = X_master['OpponentWinner'].astype(int)
    
    inds = X_master[X_master['streakType'] == '-1'].index
    X_master.loc[inds,'streakType'] = -1
    inds = X_master[X_master['teamId'] == '-1'].index
    X_master.loc[inds,'teamId'] = -1
    inds = X_master[X_master['OpponentId'] == '-1'].index
    X_master.loc[inds,'OpponentId'] = -1
    inds = X_master[X_master['divisionId'] == '-1'].index
    X_master.loc[inds,'divisionId'] = -1
    inds = X_master[X_master['toTeamId'] == '-1'].index
    X_master.loc[inds,'toTeamId'] = -1    
    inds = X_master[X_master['toTeamId'] == '0'].index
    X_master.loc[inds,'toTeamId'] = -1        
    inds = X_master[X_master['toTeamId'] == 0].index
    X_master.loc[inds,'toTeamId'] = -1        
    
    X_master = team_clustering(X_master,team_clusters)
    
    #One hot encode several categorial variables    
    X_master = pd.get_dummies(X_master,columns=['teamId_payrollCluster','teamId_attendCluster',
                                                'streakType','divisionId',
                                                'OpponentId_payrollCluster','OpponentId_attendCluster',
                                                'toTeamId_payrollCluster','toTeamId_attendCluster'])
    
    #If instance of expected one hot encoding not present, add in
    expected_encodings = set(['divisionId_-1','divisionId_200','divisionId_201','divisionId_202','divisionId_203',
                             'divisionId_204','divisionId_205', 'streakType_-1','streakType_l','streakType_w',
                             'teamId_payrollCluster_other','teamId_payrollCluster_-1','teamId_payrollCluster_0',
                             'teamId_payrollCluster_1','teamId_payrollCluster_2','teamId_payrollCluster_3',
                             'teamId_payrollCluster_4','teamId_attendCluster_other','teamId_attendCluster_-1',
                             'teamId_attendCluster_0','teamId_attendCluster_1','teamId_attendCluster_2',
                             'teamId_attendCluster_3','teamId_attendCluster_4','OpponentId_payrollCluster_other',
                             'OpponentId_payrollCluster_-1','OpponentId_payrollCluster_0','OpponentId_payrollCluster_1',
                             'OpponentId_payrollCluster_2','OpponentId_payrollCluster_3','OpponentId_payrollCluster_4',
                             'OpponentId_attendCluster_other',
                             'OpponentId_attendCluster_-1','OpponentId_attendCluster_0','OpponentId_attendCluster_1',
                             'OpponentId_attendCluster_2','OpponentId_attendCluster_3','OpponentId_attendCluster_4',
                             'toTeamId_payrollCluster_other','toTeamId_payrollCluster_-1','toTeamId_payrollCluster_0',
                             'toTeamId_payrollCluster_1','toTeamId_payrollCluster_2','toTeamId_payrollCluster_3',
                             'toTeamId_payrollCluster_4','toTeamId_attendCluster_other',
                             'toTeamId_attendCluster_-1','toTeamId_attendCluster_0',
                             'toTeamId_attendCluster_1','toTeamId_attendCluster_2','toTeamId_attendCluster_3',
                             'toTeamId_attendCluster_4'])

    #If instance of expected features are not present, add in.
    for i,encoding in enumerate(expected_encodings):
        if encoding not in list(X_master.columns):
            X_master[encoding] = 0

    #Add day of week feature
    X_master['DayOfWeek'] = X_master['dailyDataDate'].dt.weekday
    
    if nextDayPlayerEngagement is not None:
        print('Merging with nextDayPlayerEngagement')
        nextDayPlayerEngagement['playerId'] = nextDayPlayerEngagement['playerId'].astype(str)
        nextDayPlayerEngagement['playerId'] = nextDayPlayerEngagement['playerId'].str.strip().str.lower()
        X_master = X_master.merge(nextDayPlayerEngagement,how='outer',
                                  left_on=['dailyDataDate','playerId'],right_on=['dailyDataDate','playerId'])
        print('X_master shape: ',X_master.shape)
        print('X_master.isna().any().any(): ',X_master.isna().any().any())
    else:
        X_master[['engagementMetricsDate','target1','target2','target3','target4']] = -1
        
    final_order = ['dailyDataDate','playerId','mlbDebutDate_DaysRelative','heightInches','weight','BornInUS?','primaryPositionCode_1',
                   'primaryPositionCode_10','primaryPositionCode_2','primaryPositionCode_3','primaryPositionCode_4','primaryPositionCode_5',
                   'primaryPositionCode_6','primaryPositionCode_7',
                   'primaryPositionCode_8','primaryPositionCode_9','primaryPositionCode_i','primaryPositionCode_o',
                   'statusCode_a','statusCode_dl','statusCode_other',
                   'statusCode_rm','numGames','gamePk','gamePk2','atBats','plateAppearances',
                   'hits','homeRuns','doubles','triples','baseOnBalls','strikeOuts',
                   'intentionalWalks','hitByPitch','rbi','leftOnBase','sacBunts','sacFlies','flyOuts',
                   'groundOuts','stolenBases','runsScored','totalBases',
                   'caughtStealing','pickoffs','errors','chances','putOuts','assists',
                   'caughtStealingPitching','gamesPlayedBatting','gamesPlayedPitching',
                   'gamesStartedPitching','completeGamesPitching','shutoutsPitching','winsPitching',
                   'saveOpportunities','saves','blownSaves','strikeOutsPitching',
                   'battersFaced','strikes','balls','pitchesThrown','wildPitches','earnedRuns','noHitter',
                   'pitchingGameScore','inningsPitchedAsFrac',
                   'positionName_playedInGame_catcher','positionName_playedInGame_designated hitter','positionName_playedInGame_first base',
                   'positionName_playedInGame_outfielder','positionName_playedInGame_pinch hitter','positionName_playedInGame_pinch runner',
                   'positionName_playedInGame_pitcher','positionName_playedInGame_second base',
                   'positionName_playedInGame_shortstop','positionName_playedInGame_third base','n_PTFollowers',
                   'n_TTFollowers','toDateMinorLeagueAccolades','toDateMVPs','toDateCyYoungs',
                   'toDateRookieOfYears','toDateSilverSluggers','toDateWorldSeriesChampionships','toDateWorldSeriesMVPs',
                   'toDateLCSMVPs','toDatePlayerOfWeeks','toDatePlayerOfMonths','toDateAllStars','year','month','inSeason','AllStarGame',
                   'seasonPart_All-Star_Break','seasonPart_Between_Reg_and_Postseason','seasonPart_Offseason','seasonPart_Postseason',
                   'seasonPart_Preseason','seasonPart_Reg_Season_1st_Half','seasonPart_Reg_Season_2nd_Half','gameTimeUTC','TeamHits',
                   'TeamDoubles','TeamTriples','TeamHomeRuns','TeamStrikeOuts','TeamBaseOnBalls','TeamIntentionalWalks','TeamHBP',
                   'TeamRunsScored','TeamStolenBases','TeamLeftOnBase','TeamStrikeOutsPitching','HomeTeam','TeamWinPct','OpponentWinPct',
                   'TeamWins','OpponentWins','TeamLosses','OpponentLosses','TeamScore','OpponentScore','TeamWinner','OpponentWinner',
                   'divisionRank','leagueRank','wildCardRank','divisionGamesBack','leagueGamesBack','wins','losses','homeWins','awayWins',
                   'extraInningWins','oneRunWins','eliminationNumber','wildCardEliminationNumber',
                   'divisionChamp','divisionLeader','wildCardLeader',
                   'streakValue','pitches100mph','HRDist450ft','gameTyingRBI','goAheadRBI',
                   'walkoffRBI','assigned','claimedOffWaivers','declaredFreeAgency',
                   'designatedforAssignment','optioned','recalled','released','selected','signed','signedasFreeAgent','statusChange','trade',
                   'teamId','divisionId_-1','divisionId_200','divisionId_201','divisionId_202','divisionId_203',
                   'divisionId_204','divisionId_205', 'streakType_-1','streakType_l','streakType_w',
                   'teamId_payrollCluster_other','teamId_payrollCluster_-1','teamId_payrollCluster_0',
                   'teamId_payrollCluster_1','teamId_payrollCluster_2','teamId_payrollCluster_3',
                   'teamId_payrollCluster_4','teamId_attendCluster_other','teamId_attendCluster_-1',
                   'teamId_attendCluster_0','teamId_attendCluster_1','teamId_attendCluster_2',
                   'teamId_attendCluster_3','teamId_attendCluster_4','OpponentId_payrollCluster_other',
                   'OpponentId_payrollCluster_-1','OpponentId_payrollCluster_0','OpponentId_payrollCluster_1',
                   'OpponentId_payrollCluster_2','OpponentId_payrollCluster_3','OpponentId_payrollCluster_4',
                   'OpponentId_attendCluster_other',
                   'OpponentId_attendCluster_-1','OpponentId_attendCluster_0','OpponentId_attendCluster_1',
                   'OpponentId_attendCluster_2','OpponentId_attendCluster_3','OpponentId_attendCluster_4',
                   'toTeamId_payrollCluster_other','toTeamId_payrollCluster_-1','toTeamId_payrollCluster_0',
                   'toTeamId_payrollCluster_1','toTeamId_payrollCluster_2','toTeamId_payrollCluster_3',
                   'toTeamId_payrollCluster_4','toTeamId_attendCluster_other',
                   'toTeamId_attendCluster_-1','toTeamId_attendCluster_0',
                   'toTeamId_attendCluster_1','toTeamId_attendCluster_2','toTeamId_attendCluster_3',
                   'toTeamId_attendCluster_4','DayOfWeek',
                   'engagementMetricsDate','target1','target2','target3','target4']
    X_master = X_master.loc[:,final_order]
    X_master.reset_index(drop=True,inplace=True)

    return X_master

def create_X_master(unpacked_dfs, players, teams, seasons, awards_static, team_clusters,
                                                            training_master=None, Y=None):
    #Assign variables to the unpacked dfs, and get player_ids and dailyDataDates for later use
    if Y is not None:
        games, rosters, playerBoxScores, teamBoxScores, \
               transactions, standings, awards, events, \
               playerTwitterFollowers, teamTwitterFollowers \
               = assign_vars(unpacked_dfs,Y)
        player_ids = set(Y['playerId'].astype(str).str.strip().str.lower().values)
        dailyDataDates = sorted(list(Y['date'].unique()))
        #Extract training_master data closest to dailyDataDate
        training_master.set_index('dailyDataDate',inplace=True)
        training_master.index = pd.to_datetime(training_master.index)
        training_master.sort_index(inplace=True)
        try:
            ind = training_master.index.get_loc(dailyDataDates[0])
            #print(ind)
            #print(training_master.iloc[ind,:])
            #print(training_master)
            training_master = training_master.iloc[ind,:]
        except:
            ind = training_master.index[-1]
            #print(ind)
            #print(training_master.loc[ind,:])
            #print(training_master)
            training_master = training_master.loc[ind,:]
        training_master.reset_index(inplace=True)
        #print(training_master)
    else:
        nextDayPlayerEngagement, games, rosters, playerBoxScores, teamBoxScores, \
                                  transactions, standings, awards, events, \
                                  playerTwitterFollowers, teamTwitterFollowers\
                                  = assign_vars(unpacked_dfs,Y)
        player_ids = set(nextDayPlayerEngagement['playerId'].astype(str).str.
                     strip().str.lower().unique())
        dailyDataDates = sorted(list(nextDayPlayerEngagement['dailyDataDate'].unique()))
    
    #Print player_ids being tracked and daily data dates
    print(len(player_ids))
    print(len(dailyDataDates))
    games2 = games.copy()
    
    #Get team ID's that will be tracked (major league teams and 'other')
    team_ids = set(list(teams['id'].astype(str).str.strip(). 
                        str.lower().values) + ['other','-1']) #,'159','160'])
    print(team_ids)
    
    #Make Roster info ************************************
    print('****************************************************************')
    print('Processing roster dataframe...') 
    float_features = []
    bool_features = []
    cat_features = ['playerId','teamId','statusCode']
    date_features = ['dailyDataDate']
    #print('# of unique dates in rosters: ',len(rosters['dailyDataDate'].unique()))
    #print('# of unique players in rosters: ',len(rosters['playerId'].unique()))
    X_rosters = make_rosters(rosters, date_features, float_features, bool_features, 
                             cat_features, player_ids, team_ids)
    print('X_rosters: ', X_rosters)
    #print('# of unique dates in X_rosters: ',len(X_rosters['dailyDataDate'].unique()))
    #print('# of unique players in X_rosters: ',len(X_rosters['playerId'].unique()))
    print('X_rosters.isna().any().any(): ',X_rosters.isna().any().any())    
    print('Which column(s) is NaN?: ',X_rosters.isna().any()[X_rosters.isna().any() == True])
    print('X_rosters shape: ',X_rosters.shape)
    print('Roster dataframe processing complete.')
    
    #Make Player info ************************************
    print('****************************************************************')
    print('Processing player info dataframe...')
    float_features = ['heightInches','weight']
    bool_features = []
    cat_features = ['playerId','primaryPositionCode','birthCountry']
    date_features = ['mlbDebutDate']
    X_playerInfo = make_playerInfo(players, date_features, float_features, bool_features, 
                                 cat_features, player_ids, dailyDataDates)
    print('playerInfo: ', X_playerInfo)
    print('X_playerInfo.isna().any().any(): ',X_playerInfo.isna().any().any())
    print('Which column(s) is NaN?: ',X_playerInfo.isna().any()[X_playerInfo.isna().any() == True])
    print('Player info shape: ', X_playerInfo.shape)
    #print('Player info # of unique dates: ',len(X_playerInfo['dailyDataDate'].unique()))
    #print('Player info # of unique players: ',len(X_playerInfo['playerId'].unique()))
    #print('Player info processing complete.')

    #Merge roster and player info datasets, and handle NaNs
    print('****************************************************************')
    X_master = merge_rosters_playerInfo_process_NaNs(X_rosters, X_playerInfo, training_master)
    print('X_master: ', X_master)

    #Player box scores ************************************
    print('****************************************************************')
    print('Processing player box scores dataframe...') 
    float_features = ['atBats',   
                      'plateAppearances',
                      'hits',
                      'homeRuns',
                      'doubles',
                      'triples',
                      'baseOnBalls',
                      'strikeOuts',
                      'intentionalWalks',
                      'hitByPitch',
                      'rbi',
                      'leftOnBase',
                      'sacBunts',
                      'sacFlies',
                      'flyOuts',
                      'groundOuts',
                      'stolenBases', 
                      'runsScored',
                      'totalBases',
                      'caughtStealing', 
                      'pickoffs',
                      'errors', 
                      'chances',
                      'putOuts',
                      'assists',
                      'caughtStealingPitching', 
                      'gamesPlayedBatting', 
                      'gamesPlayedPitching',
                      'gamesStartedPitching',
                      'completeGamesPitching',
                      'shutoutsPitching',
                      'outsPitching',
                      'winsPitching',
                      'saveOpportunities',
                      'saves',
                      'blownSaves',
                      'strikeOutsPitching',
                      'inningsPitched',
                      'battersFaced',
                      'strikes',
                      'balls',
                      'pitchesThrown',
                      'wildPitches',
                      'runsPitching',
                      'homeRunsPitching',
                      'hitsPitching',
                      'baseOnBallsPitching',
                      'earnedRuns']
    bool_features = []
    cat_features = ['playerId','teamId','gamePk','positionName']
    date_features = ['dailyDataDate']
    X_playerBoxScores = make_playerBoxScores(playerBoxScores, date_features, float_features,
                                             bool_features, cat_features, player_ids, team_ids)
    #print('# of unique dates player box scores: ',len(X_playerBoxScores['dailyDataDate'].unique()))
    #print('# of unique players player box scores: ',len(X_playerBoxScores['playerId'].unique()))
    #print('# of unique teams player box scores: ',len(X_playerBoxScores['teamId'].unique()))
    print('X_playerBoxScores: ',X_playerBoxScores)
    print('X_playerBoxScores shape: ',X_playerBoxScores.shape)
    print('X_playerBoxScores.isna().any().any(): ',X_playerBoxScores.isna().any().any())
    print('PlayerBoxScore processing complete.') 
    #pdb.set_trace()    
    #Clear space
    del(rosters,X_playerInfo,X_rosters)
    
    #Merge X_master and player box scores
    print('****************************************************************')
    X_master = merge_master_playerBoxScores_process_NaNs(X_master, X_playerBoxScores, training_master)
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True])
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    print('X_master: ', X_master)
    del playerBoxScores
    del X_playerBoxScores
    #raise SystemExit

    
    #Player twitters ************************************
    print('****************************************************************')
    print('Processing player twitter dataframe...')     
    float_features = ['numberOfFollowers']
    bool_features = []
    cat_features = ['playerId']
    date_features = ['dailyDataDate']
    X_playerTwitters = make_playerTwitters(playerTwitterFollowers, date_features, float_features,
                                           bool_features,cat_features, player_ids)
    print('X_playerTwitters: ', X_playerTwitters)
    print('X_playerTwitters shape: ',X_playerTwitters.shape)
    #print('# of unique dates playerTwitters: ',len(X_playerTwitters['dailyDataDate'].unique()))
    #print('# of unique players playerTwitters: ',len(X_playerTwitters['playerId'].unique()))
    
    print('****************************************************************')
    X_master = merge_master_playerTwitters_process_NaNs(X_master, X_playerTwitters, training_master)
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True])
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    print('X_master: ', X_master)
    
    #Team twitters ************************************
    print('****************************************************************')
    print('Processing team twitter dataframe...')    
    float_features = ['numberOfFollowers']
    bool_features = []
    cat_features = ['teamId']
    date_features = ['dailyDataDate']
    X_teamTwitters = make_teamTwitters(teamTwitterFollowers, date_features, float_features,
                                       bool_features, cat_features, team_ids)
    print('X_teamTwitters: ', X_teamTwitters)
    print('X_teamTwitters shape: ',X_teamTwitters.shape)
    #print('# of unique dates, teamTwitters: ',len(X_teamTwitters['dailyDataDate'].unique()))
    #print('# of unique teams, teamTwitters: ',len(X_teamTwitters['teamId'].unique()))
    
    print('****************************************************************')
    X_master = merge_master_teamTwitters_process_NaNs(X_master, X_teamTwitters, training_master)
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True])
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    print('X_master: ', X_master)
    del(playerTwitterFollowers, teamTwitterFollowers, X_playerTwitters, X_teamTwitters)
    
    #Awards ************************************
    print('****************************************************************')
    print('Processing award dataframe...')    
    X_awards = make_awards(awards, awards_static, player_ids)
    print('# of unique dates, X_awards: ',len(X_awards['dailyDataDate'].unique()))
    print('# of unique players, X_awards: ',len(X_awards['playerId'].unique()))
    print('X_awards shape: ',X_awards.shape)
    print('X_awards: ',X_awards)
    
    print('****************************************************************')
    X_master = merge_master_awards_process_NaNs(X_master, X_awards, training_master)
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True])
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    print('X_master: ', X_master)
    del awards
    del players

    #Season info ************************************
    print('****************************************************************')
    print('Processing season dataframe...')    
    dates_with_season_part = make_seasons(seasons, X_master)
    print('dates_with_season_part shape: ',dates_with_season_part.shape)
    print('dates_with_season_part: ',dates_with_season_part)

    print('****************************************************************')
    print('Merging X_master with seasons...')
    print('X_master shape: ',X_master.shape)
    #Season. swill always exist, and with no NaNs, so no need for additional function
    X_master = X_master.merge(dates_with_season_part, how='outer', left_on = ['dailyDataDate'],
                                                                   right_on = ['dailyDataDate'])
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True])
    print('X_master shape: ',X_master.shape)
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    print('X_master: ', X_master)


    #Team box scores ************************************
    print('****************************************************************')
    print('Processing team box scores dataframe...')    
    float_features = ["hits",
        "doubles",
        "triples",
        "homeRuns",
        "strikeOuts",
        "baseOnBalls",
        'intentionalWalks',
        'hitByPitch',
        "runsScored",
        "stolenBases",
        'leftOnBase',
        "strikeOutsPitching",
        "home"]
    bool_features = []
    cat_features = ['teamId','gamePk']
    date_features = ['dailyDataDate','gameTimeUTC']
    X_teamBoxScores = make_teamBoxScores(teamBoxScores,date_features,float_features,
                                         bool_features,cat_features,team_ids)
    #print('# of unique dates, team box scores: ',len(X_teamBoxScores['dailyDataDate'].unique()))
    #print('# of unique teams, team box scores: ',len(X_teamBoxScores['teamId'].unique()))
    #print('# of unique games, team box scores: ', len(X_teamBoxScores['gamePk'].unique()))
    print('X_teamBoxScores shape: ',X_teamBoxScores.shape)
    print('X_teamBoxScores: ',X_teamBoxScores)
    del teamBoxScores
    gc.collect()

    print('****************************************************************')
    X_master = merge_master_teamBoxScores_process_NaNs(X_master, X_teamBoxScores)
    print('X_master shape: ',X_master.shape)
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True])  
    print('X_master: ', X_master)

    #Game info ************************************
    print('****************************************************************')
    print('Processing game info dataframe...')    
    float_features = ['homeWinPct',
        'awayWinPct',
        'homeWins',
        'homeLosses',
        'homeScore',
        'awayWins',
        'awayLosses',
        'awayScore']
    bool_features = ['homeWinner','awayWinner']
    cat_features = ['gamePk','homeId','awayId']
    date_features = ['dailyDataDate']
    X_gameInfo = make_gameInfo(games, date_features, float_features, bool_features, 
                               cat_features, team_ids)
    #print('# of unique dates, X_gameinfo: ',len(X_gameInfo['dailyDataDate'].unique()))
    #print('# of unique teams, X_gameinfo: ',len(set(X_gameInfo['awayId'].unique()).union(set(X_gameInfo['homeId'].unique()))))
    #print('# of unique games, X_gameinfo: ', len(X_gameInfo['gamePk'].unique()))
    print('X_gameinfo shape: ',X_gameInfo.shape)
    print('X_gameinfo : ',X_gameInfo)

    print('****************************************************************')
    print(X_gameInfo.info())
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    X_master = merge_master_gameInfo_process_NaNs(X_master, X_gameInfo)
    print('X_master shape: ',X_master.shape)
    print('X_master.isna().any().any(): ',X_master.isna().any().any())    
    print('X_master: ', X_master)
    
    print('****************************************************************')
    print('Additional feature engineering...')        
    #X_master = encode_home_away(X_master,'AwayTeam')
    X_master = encode_opponent(X_master,'OpponentId','homeId','awayId')
    X_master = encode_team(X_master,'TeamWinPct','homeWinPct','awayWinPct')
    X_master = encode_opponent(X_master,'OpponentWinPct','homeWinPct','awayWinPct')
    X_master = encode_team(X_master,'TeamWins','homeWins','awayWins')
    X_master = encode_opponent(X_master,'OpponentWins','homeWins','awayWins')
    X_master = encode_team(X_master,'TeamLosses','homeLosses','awayLosses')
    X_master = encode_opponent(X_master,'OpponentLosses','homeLosses','awayLosses')
    X_master = encode_team(X_master,'TeamScore','homeScore','awayScore')
    X_master = encode_opponent(X_master,'OpponentScore','homeScore','awayScore')
    X_master = encode_team(X_master,'TeamWinner','homeWinner','awayWinner')
    X_master = encode_opponent(X_master,'OpponentWinner','homeWinner','awayWinner')
    X_master.drop(columns=['homeWinPct','awayWinPct','homeWins','homeLosses',
                           'homeScore','awayWins','awayLosses','awayScore','homeWinner',
                           'awayWinner','homeId','awayId'],inplace=True)
    print('Additional feature engineering complete.')        
    print('X_master shape: ',X_master.shape)
    print('X_master.isna().any().any(): ',X_master.isna().any().any())    
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True])
    print('X_master: ', X_master)
      
        
    #Standings ************************************
    print('****************************************************************')
    print('Processing standings dataframe...')            
    float_features = ['divisionRank','leagueRank','wildCardRank','divisionGamesBack','leagueGamesBack',
                      'wins','losses','homeWins','awayWins','extraInningWins',
                      'oneRunWins','eliminationNumber','wildCardEliminationNumber']
    bool_features = ['divisionChamp','divisionLeader','wildCardLeader']
    cat_features = ['streakCode','teamId','divisionId']
    date_features = ['dailyDataDate']
    X_standings = make_standings_info(standings, cat_features, float_features, date_features, 
                                      bool_features, team_ids)
    print('X_standings shape: ',X_standings.shape)
    print('X_standings : ',X_standings)
    
    print('****************************************************************')
    X_master = merge_master_standings_process_NaNs(X_master, X_standings, training_master)
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True]) 
    print('X_master: ',X_master)
    del X_gameInfo
    del X_teamBoxScores
    gc.collect()
    
    #Events ************************************
    print('****************************************************************')
    print('Processing events dataframe...')      
    float_features = []
    bool_features = []
    cat_features = ['gamePk','playerId']
    X_events = make_events_info(events, games2,cat_features, float_features, 
                                bool_features, player_ids)
    print(X_events.info())
    print('X_events shape: ',X_events.shape)
    
    
    print('****************************************************************')
    #Note that number. ofrows increases after merge due to their being multiple homerun events for a plyerID pergame
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    X_master = merge_master_events_process_NaNs(X_master, X_events)
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True]) 
    print('X_master: ',X_master)
    del events
    del standings
    del games

    #Transactions ************************************
    print('****************************************************************')
    print('Processing transactions dataframe...')      
    X_transactions = make_transactions_info(transactions, player_ids, team_ids)
    print(X_transactions.info())
    print('X_transactions shape: ',X_transactions.shape)
    
    
    print('****************************************************************')
    #Note that number. ofrows increases after merge due to their being multiple homerun events for a plyerID pergame
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    X_master = merge_master_transactions_process_NaNs(X_master, X_transactions)
    print(X_master[['dailyDataDate','gamePk','gamePk2']].info())
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True]) 
    print('X_master: ',X_master)    
    
    #Do some final processing of features matrix
    if Y is None:
        X_master = final_processing(X_master, team_clusters, nextDayPlayerEngagement)
        del nextDayPlayerEngagement
    else:
        X_master = final_processing(X_master, team_clusters)
    print('X_master.isna().any().any(): ',X_master.isna().any().any())
    print('Which column(s) is NaN?: ',X_master.isna().any()[X_master.isna().any() == True]) 
    print('X_master: ',X_master)

    return X_master

def make_predictions(training_master,sample_predict_df,models):
    models1 = models[0]
    models2 = models[1]
    models3 = models[2]
    models4 = models[3]
    player_id_predicts = training_master['playerId'].values
    X_master = training_master.drop(columns=['teamId','gamePk','gamePk2','engagementMetricsDate','playerId',
                                             'dailyDataDate','target1','target2','target3','target4'])
    print('X_master shape for inference: ',X_master.shape)
    model_1 = xgb.XGBRegressor()
    model_1.load_model(models1)
    
    model_2 = xgb.XGBRegressor()
    model_2.load_model(models2)
    
    model_3 = xgb.XGBRegressor()
    model_3.load_model(models3)
    
    model_4 = xgb.XGBRegressor()
    model_4.load_model(models4)    
    
    player_id_prescribed_order = sample_predict_df['playerId'].values
    sample_predict_df.set_index('date',inplace=True)
    sample_predict_df.drop(columns=['DatePrediction','playerId'],inplace=True)
    
    y1 = pd.Series(data = model_1.predict(X_master),index = player_id_predicts).clip(0,100)
    y1 = y1.groupby(player_id_predicts).mean().reindex(player_id_prescribed_order)

    y2 = pd.Series(data = model_2.predict(X_master),index = player_id_predicts).clip(0,100)
    y2 = y2.groupby(player_id_predicts).mean().reindex(player_id_prescribed_order)
    
    y3 = pd.Series(data = model_3.predict(X_master),index = player_id_predicts).clip(0,100)
    y3 = y3.groupby(player_id_predicts).mean().reindex(player_id_prescribed_order)
    
    y4 = pd.Series(data = model_4.predict(X_master),index = player_id_predicts).clip(0,100)
    y4 = y4.groupby(player_id_predicts).mean().reindex(player_id_prescribed_order)

    sample_predict_df['target1'] = y1.values
    sample_predict_df['target2'] = y2.values
    sample_predict_df['target3'] = y3.values
    sample_predict_df['target4'] = y4.values
    
    return sample_predict_df