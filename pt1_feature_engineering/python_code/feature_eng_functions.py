import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
import json

def load_json(filename):
        '''
    Function: 
        - opens json file and store information in a pandas dataframe
        - also prints out aggregated df with counts of picture by countrycode
    Input:
        1. filename/path ex: ./data/filename.json
    Output:
        1. new dataframe containing json info
    '''
    df = pd.read_json(filename, lines=True)
    test = df.groupby(df['countrycode']).count()
    print test.sort(columns='drawing',ascending=False).head(15)
    return df

def pic_viewer(df_cf, _id):
    
    '''
    Function: 
        - If X and Y columns exist in your dataframe, you can use this function to view drawing with specific id.
    Input:
        1. dataframe df_cf
        2. object id _id
    Output:
        1. scatter plot of x and y
    '''
    plt.scatter(df_cf.X[_id],df_cf.Y[_id])
    plt.gca().invert_yaxis()

    
def feature_eng_pt0(df, countries = ['US','BR','RU','KR'], limit = 3500):
        '''
    Function: 
        - takes dataframe and filter by countries that are in a list called countries.
        - number of datapoints per country is limited by a integer called limit
        - for instance, if limit = 3000 and countries = ['US','DE','GB'], 
        output dataframe contains max 9000 datapoints from US,Germany and Great Britain (3000 max each).
        - also, I decided to extract drawings with less than 30 strokes.
        
    Input:
        1. dataframe df
        2. countries [list]
        3. limit [int]
    Output:
        1. dataframe containing 'limit' * len(countries) rows
    '''
    
    #filter dataframe with selected countries
    df_cf = df[(df['countrycode']==countries[0])|(df['countrycode']==countries[1])|\
               (df['countrycode']==countries[2])|(df['countrycode']==countries[3])]
    
    # create feature "stroke_number"
    df_cf['stroke_number']=df_cf['drawing'].str.len()
    
    
    # US
    df_us = df_cf[(df_cf['countrycode']=='US') & (df_cf['stroke_number'] <= 30)]
    random_US = np.random.choice(list(df_us.index), limit, replace=False)
    df_US = df_us.loc[list(random_US)]
    
    #BR
    if df[df['countrycode']=='BR'].count()[0] > limit:
        df_BR = df_cf[(df_cf['countrycode']=='BR') & (df_cf['stroke_number'] <= 30)]
        random_BR = np.random.choice(list(df_BR.index), limit, replace=False)
        df_BR = df_BR.loc[list(random_BR)]
    else: 
        df_BR = df_cf[(df_cf['countrycode']=='BR') & (df_cf['stroke_number'] <= 30)]

    #RU
    if df[df['countrycode']=='RU'].count()[0] > limit:
        df_RU = df_cf[(df_cf['countrycode']=='RU') & (df_cf['stroke_number'] <= 30)]
        random_RU = np.random.choice(list(df_RU.index), limit, replace=False)
        df_RU = df_RU.loc[list(random_RU)]
    else: 
        df_RU = df_cf[(df_cf['countrycode']=='RU') & (df_cf['stroke_number'] <= 30)]

    #KR
    if df[df['countrycode']=='KR'].count()[0] > limit:
        df_KR = df_cf[(df_cf['countrycode']=='KR') & (df_cf['stroke_number'] <= 30)]
        random_KR = np.random.choice(list(df_KR.index), limit, replace=False)
        df_KR = df_KR.loc[list(random_KR)]
    else: 
        df_KR = df_cf[(df_cf['countrycode']=='KR') & (df_cf['stroke_number'] <= 30)]
        
    return pd.concat([df_US,df_BR,df_RU,df_KR], axis=0)


def feature_eng_pt1(df_cf):    
    #create "final_time","first_X","first_Y"
    final_time = []
    beginningX = []
    beginningY = []
    for i in df_cf.index:
        num = df_cf.stroke_number[i]
        final_time.append(df_cf.loc[i,'drawing'][num-1][2][-1])
        beginningX.append(df_cf.loc[i,'drawing'][0][0][0])
        beginningY.append(df_cf.loc[i,'drawing'][0][1][0])
    df_cf['final_time'] = final_time
    df_cf['first_X'] = beginningX
    df_cf['first_Y'] = beginningY
    
    #create 'Year','month','day','hour_created'
    df_cf.timestamp = pd.to_datetime(df_cf.timestamp)
    df_cf['year'] = df_cf['timestamp'].dt.year
    df_cf['month'] = df_cf['timestamp'].dt.month
    df_cf['day'] = df_cf['timestamp'].dt.day
    df_cf['hour_created'] = df_cf['timestamp'].dt.hour
    
    b_loon = {True: 1, False:0}
    df_cf['recognized'] = df_cf['recognized'].map(b_loon)
    return df_cf

def feature_eng_pt2(df_cf):
    X = {}
    Xtemp =[]
    Y = {}
    Ytemp =[]
    time = {}
    ttemp =[]
    Tdiff = {}
    Tdifftemp = []
    ttnum_dp = {}
    Tdiffmax = {}
    Tdiffmin = {}
    Tdiffstd = {}
    dpps_temp = []
    dpps = {}
    dp_max = {}
    dp_min = {}
    dp_std = {}
    sumtimeps = {}

    for i in df_cf.index:
        num = df_cf.stroke_number[i]
        for stroke in xrange(num):
            Xtemp += (df_cf.loc[i,'drawing'][stroke][0])
            Ytemp += (df_cf.loc[i,'drawing'][stroke][1])
            ttemp += (df_cf.loc[i,'drawing'][stroke][2])
            Tdifftemp.append(df_cf.loc[i,'drawing'][stroke][2][-1] - df_cf.loc[i,'drawing'][stroke][2][0])
            dpps_temp.append(len(df_cf.loc[i,'drawing'][stroke][2]))
        X[i] = Xtemp
        Y[i] = Ytemp
        time[i] = ttemp
        ttnum_dp[i] = len(Xtemp)
        Tdiff[i] = Tdifftemp
        Tdiffmax[i] = np.argmax(Tdifftemp,axis=0)
        Tdiffmin[i] = np.argmin(Tdifftemp,axis=0)
        Tdiffstd[i] = np.std(Tdifftemp)
        dpps[i] = dpps_temp
        dp_max[i] = np.argmax(dpps_temp,axis=0)
        dp_min[i] = np.argmin(dpps_temp,axis=0)
        dp_std[i] = np.std(dpps_temp)
        sumtimeps[i] = sum(Tdifftemp) 
        
        Tdifftemp=[]
        Xtemp = []
        Ytemp = []
        ttemp = []
        dpps_temp = []
    df_cf['total_number_of_datapoints'] = pd.Series(ttnum_dp)
    df_cf['X'] = pd.Series(X)
    df_cf['Y'] = pd.Series(Y)
    df_cf['time'] = pd.Series(time)
    df_cf['time_per_stroke'] = pd.Series(Tdiff)
    df_cf['dp_per_stroke'] = pd.Series(dpps)
    df_cf['stroke_with_max_time'] = pd.Series(Tdiffmax)
    df_cf['stroke_with_min_time'] = pd.Series(Tdiffmin)
    df_cf['std_of_time'] = pd.Series(Tdiffstd)
    df_cf['ave_datapoints_per_stroke'] = df_cf['total_number_of_datapoints']/(df_cf['stroke_number'])
    df_cf['total_time_drawing'] = pd.Series(sumtimeps)
    df_cf['ave_time_per_stroke'] = df_cf['total_time_drawing']/(df_cf['stroke_number'])
    df_cf['stroke_with_max_dp'] = pd.Series(dp_max)
    df_cf['stroke_with_min_dp'] = pd.Series(dp_min)
    df_cf['std_of_dp'] = pd.Series(dp_std)
    return df_cf


def feature_eng_pt3(df_cf):
    slope = {}
    stemp =[]
    direction = {}
    directiontemp = []

    for i in df_cf.index:
        num = df_cf.stroke_number[i]
        for stroke in xrange(num):
            dy = float(df_cf.drawing[i][stroke][1][-1] - df_cf.drawing[i][stroke][1][0])
            dx = float(df_cf.drawing[i][stroke][0][-1] - df_cf.drawing[i][stroke][0][0])
            if dx != 0:
                stemp.append(dy/dx)
            else:
                dx = 0.000001
                stemp.append(dy/dx)
            if dy < 0.0 and dx > 0.0:
                directiontemp.append(2*np.pi + np.arctan(dy/dx))
            elif dy >=0.0 and dx > 0.0:
                directiontemp.append(np.arctan(dy/dx))
            else:
                directiontemp.append(np.pi + np.arctan(dy/dx))
                
        slope[i] = stemp
        direction[i] = directiontemp
        stemp = []
        directiontemp = []
    df_cf['slope'] = pd.Series(slope)
    df_cf['direction'] = pd.Series(direction)
    return df_cf

def feature_eng_pt4(df_cf):
        '''
    Function: 
        - take columns that contains list in each row and generate new columns to store values from the list
    Input:
        1. dataframe df_cf
    Output:
        1. scatter plot of x and y
    '''
    
    for i in xrange(30):
        df_cf['slope_{}'.format(i)] = 0
        df_cf['direction_{}'.format(i)] = 0
        df_cf['time_per_stroke_{}'.format(i)] = 0
        df_cf['dp_per_stroke_{}'.format(i)] = 0
    for ii in df_cf.index:
        for iii in xrange(df_cf.stroke_number[ii]):
            df_cf.loc[ii,'slope_{}'.format(iii)] = df_cf['slope'][ii][iii]
            df_cf.loc[ii,'direction_{}'.format(iii)] = df_cf['direction'][ii][iii]
            df_cf.loc[ii,'time_per_stroke_{}'.format(iii)] = df_cf['time_per_stroke'][ii][iii]
            df_cf.loc[ii,'dp_per_stroke_{}'.format(iii)] = df_cf['dp_per_stroke'][ii][iii]
    return df_cf

def feature_eng_pt_test(df_cf):
    dummies = pd.get_dummies(df_cf['countrycode'], prefix='country', drop_first=True)
    return  df_cf.join(dummies)