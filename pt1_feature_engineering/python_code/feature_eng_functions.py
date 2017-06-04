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


def feature_eng_pt0(df, countries = ['US','BR','RU','KR'], limit = 2500):
    '''
    function: filter dataframe by countries and by total number of stroke (15) and 
    select "limit" number of data points from each country 
    if total number of data points from a country exceeds "limit"
    Also note that during this filtering process, I will be filter out pictures that have over 15 strokes
    
    input: 
    df = dataframe
    countries = a list of countries
    limit = max number of data points per country
    
    output:
    dataframe that contains datapoints from selected countries
    
    '''
    
    #filter dataframe with selected countries
    df_cf = df[(df['countrycode']==countries[0])|(df['countrycode']==countries[1])|\
               (df['countrycode']==countries[2])|(df['countrycode']==countries[3])]
    
    # create feature "stroke_number"
    df_cf['stroke_number']=df_cf['drawing'].str.len()
    
    np.random.seed(32113)
    # US
    df_us = df_cf[(df_cf['countrycode']=='US') & (df_cf['stroke_number'] <= 15)]
    random_US = np.random.choice(list(df_us.index), limit, replace=False)
    df_US = df_us.loc[list(random_US)]
    
    #BR
    if df[df['countrycode']=='BR'].count()[0] > limit:
        df_BR = df_cf[(df_cf['countrycode']=='BR') & (df_cf['stroke_number'] <= 15)]
        random_BR = np.random.choice(list(df_BR.index), limit, replace=False)
        df_BR = df_BR.loc[list(random_BR)]
    else: 
        df_BR = df_cf[(df_cf['countrycode']=='BR') & (df_cf['stroke_number'] <= 15)]

    #RU
    if df[df['countrycode']=='RU'].count()[0] > limit:
        df_RU = df_cf[(df_cf['countrycode']=='RU') & (df_cf['stroke_number'] <= 15)]
        random_RU = np.random.choice(list(df_RU.index), limit, replace=False)
        df_RU = df_RU.loc[list(random_RU)]
    else: 
        df_RU = df_cf[(df_cf['countrycode']=='RU') & (df_cf['stroke_number'] <= 15)]

    #KR
    if df[df['countrycode']=='KR'].count()[0] > limit:
        df_KR = df_cf[(df_cf['countrycode']=='KR') & (df_cf['stroke_number'] <= 15)]
        random_KR = np.random.choice(list(df_KR.index), limit, replace=False)
        df_KR = df_KR.loc[list(random_KR)]
    else: 
        df_KR = df_cf[(df_cf['countrycode']=='KR') & (df_cf['stroke_number'] <= 15)]
    
    return pd.concat([df_US,df_BR,df_RU,df_KR], axis=0)

def feature_eng_pt1(df_cf):    
    
    '''
    function: 
    feature engineering pt1
    final time = contains the last data point from the time list in the drawing column[int]
    recognized = changing recognized columns to boolean [int]
    
    input: 
    df = dataframe 
    
    output:
    dataframe with more features!
    '''
    
    #create "final_time"
    df_cf['final_time'] = [df_cf.loc[i,'drawing'][df_cf.stroke_number[i]-1][2][-1] for i in df_cf.index]
    
    #create 'Year','month','day','hour_created' CAUSE DATA LEAKAGE
#     df_cf.timestamp = pd.to_datetime(df_cf.timestamp)
#     df_cf['year'] = df_cf['timestamp'].dt.year
#     df_cf['month'] = df_cf['timestamp'].dt.month
#     df_cf['day'] = df_cf['timestamp'].dt.day
#     df_cf['hour_created'] = df_cf['timestamp'].dt.hour
    
    b_loon = {True: 1, False:0}
    df_cf['recognized'] = df_cf['recognized'].map(b_loon)
    return df_cf

def feature_eng_pt2(df_cf):
    '''
    function: 
    feature engineering pt2
    X = normalized X range 0 to 1 [list]
    Xmin = minimum value of X [int]
    Xmax = maximum value of X [int]
    Y = normalized Y. normalized using original X value [list]
    Ymin = minimum value of Y [int]
    Ymax = maximum value of Y [int]
    time = time [list]
    time_per_stroke = time spent in each stroke [list]
    dp_per_stroke = total number of data points in a stroke [list]
    stroke_with_max_time = index of stroke that has longest time [int]
    stroke_with_min_time = index of stroke that has shortest time [int]
    std_of_time = standard deviation of time [float]
    ave_datapoints_per_stroke = mean data points per stroke [float]
    total_time_drawing = time spent on drawing [int]
    ave_time_per_stroke = mean time per stroke [float]
    stroke_with_max_dp = index of stroke that has most data points [int]
    stroke_with_min_dp = index of stroke that has least data points [int]
    std_of_dp = standard deviation of data point list [float]
    
    input: 
    df = dataframe 
    
    output:
    dataframe with more features!
    '''
    
    # process:
    # 1. make a list or int
    # 2. store contents of 1. in a new dictionary
    # 3. make new column in your dataframe with 2. dictionary
    
    X = {}
    Y = {}
    Xperst = {}
    Yperst = {}
    Ymax ={}
    time = {}
    Tdiff = {}
    ttnum_dp = {}
    Tdiffmax = {}
    Tdiffmin = {}
    Tdiffstd = {}
    dpps = {}
    dppps = {}
    dp_max = {}
    dp_min = {}
    dp_std = {}
    sumtimeps = {}

    for i in df_cf.index:
        num = df_cf.stroke_number[i]
        #store X,Y,time of the stroke in a temp list
        Xt = [df_cf.loc[i,'drawing'][stroke][0] for stroke in xrange(num)]
        Yt = [df_cf.loc[i,'drawing'][stroke][1] for stroke in xrange(num)]
        tt = [df_cf.loc[i,'drawing'][stroke][2] for stroke in xrange(num)]
        # calculate the difference between final and initial time of a stroke
        Tdifftemp = [(df_cf.loc[i,'drawing'][stroke][2][-1] - df_cf.loc[i,'drawing'][stroke][2][0])\
                     for stroke in xrange(num)]
        # calculate the lengh of the stroke list
        dpps_temp = [len(df_cf.loc[i,'drawing'][stroke][2]) for stroke in xrange(num)]
        # normalizing X and Y
        Xtemp = [item for stroke in Xt for item in stroke]
        Ytemp = [item for stroke in Xt for item in stroke]
        time[i] = [item for stroke in Xt for item in stroke]
        
        #normalizing X and Y 
        Xmintemp = np.min(Xtemp)-1
        Xmaxtemp = np.max(Xtemp)+1
        Ymintemp = np.min(Ytemp)-1
        Xnorm = _array_normalizer(Xtemp, Xmintemp,Xmaxtemp,Xmintemp)
        Ynorm = _array_normalizer(Ytemp, Xmintemp,Xmaxtemp,Ymintemp)
        Ymax[i] = np.max(Ynorm)
        X[i] = Xnorm
        Y[i] = Ynorm
        Xperst[i] = [list(_array_normalizer(Xt[stroke],Xmintemp,Xmaxtemp,Xmintemp)) for stroke in xrange(len(Xt))]
        Yperst[i] = [list(_array_normalizer(Yt[stroke],Xmintemp,Xmaxtemp,Ymintemp)) for stroke in xrange(len(Yt))]
    
        ttnum_dp[i] = len(Xnorm)
        Tdiff[i] = Tdifftemp
        Tdiffmax[i] = np.argmax(Tdifftemp)
        Tdiffmin[i] = np.argmin(Tdifftemp)
        Tdiffstd[i] = np.std(Tdifftemp)
        dpps[i] = dpps_temp
        dppps[i] = np.array(dpps_temp)/float(len(Xtemp))
        dp_max[i] = np.argmax(dpps_temp)
        dp_min[i] = np.argmin(dpps_temp)
        dp_std[i] = np.std(dpps_temp)
        sumtimeps[i] = sum(Tdifftemp) 
    # create new features   
    df_cf['total_number_of_datapoints'] = pd.Series(ttnum_dp)
    df_cf['X'] = pd.Series(X)
    df_cf['Y'] = pd.Series(Y)
    df_cf['Ymax'] = pd.Series(Ymax)
    df_cf['time'] = pd.Series(time)
    df_cf['time_per_stroke'] = pd.Series(Tdiff)
    df_cf['dp_per_stroke'] = pd.Series(dpps)
    df_cf['dp_percent_per_stroke'] = pd.Series(dppps)
    df_cf['stroke_with_max_time'] = pd.Series(Tdiffmax)
    df_cf['stroke_with_min_time'] = pd.Series(Tdiffmin)
    df_cf['std_of_time'] = pd.Series(Tdiffstd)
    df_cf['ave_datapoints_per_stroke'] = df_cf['total_number_of_datapoints']/(df_cf['stroke_number'])
    df_cf['total_time_drawing'] = pd.Series(sumtimeps)
    df_cf['ave_time_per_stroke'] = df_cf['total_time_drawing']/(df_cf['stroke_number'])
    df_cf['stroke_with_max_dp'] = pd.Series(dp_max)
    df_cf['stroke_with_min_dp'] = pd.Series(dp_min)
    df_cf['X_per_stroke'] = pd.Series(Xperst)
    df_cf['Y_per_stroke'] = pd.Series(Yperst)
    df_cf['std_of_dp'] = pd.Series(dp_std)
    return df_cf,Xperst,Yperst

def feature_eng_pt3(df_cf):
    '''
    function: 
    feature engineering pt3
    direction = radian direction of a stroke [list]
    Calculates dx and dy. based on the sign of dx and dy it will return direction in radian.(range 0 to 6.28)
    
    input: 
    df = dataframe 
    
    output:
    dataframe with direction features!
    '''
    
    direction = {}
    for i in df_cf.index:
        dx = [float(df_cf.drawing[i][stroke][1][-1] - df_cf.drawing[i][stroke][1][0]) \
          for stroke in xrange(df_cf.stroke_number[i])]
        dy = [float(df_cf.drawing[i][stroke][0][-1] - df_cf.drawing[i][stroke][0][0]) \
          for stroke in xrange(df_cf.stroke_number[i])]
        dx = np.array(dx)
        dy = np.array(dy)
        dx[dx==0] = 0.000001
        vecrad_direction = np.vectorize(_radian_direction)
        direction[i] = vecrad_direction(dy,dx)
    df_cf['direction'] = pd.Series(direction)
    return df_cf

def feature_eng_pt4(df_cf):
    '''
    function: 
    feature engineering pt4
    generate features based on information from a stroke (x,y,t).
    will generate following features for first 15 strokes of a drawing.
    
    1.datapoint_percentage_stroke{i} = total# of data points from stroke{i} of a drawing / total# of data points of that drawing
    2.direction_stroke{i} = direction of the stroke{i} in radian (0:6.28)
    3.time_stroke{i} = total time spent on stroke{i}
    4.datapoint_stroke{i} = total number of data points from stroke{i}
    
    input: 
    df = dataframe 
    
    output:
    dataframe with 4 more features features!
    '''
    
    #first create new columns. 15*4 = 60 columns total
    for i in xrange(15):
        df_cf['datapoint_percentage_stroke{}'.format(i)] = 0
        df_cf['direction_stroke{}'.format(i)] = 0
        df_cf['time_stroke{}'.format(i)] = 0
        df_cf['datapoint_stroke{}'.format(i)] = 0
    
    #vectorize _percentfier function    
    percentfierv =np.vectorize(_percentfier)    
       
    for index in df_cf.index:
        if df_cf.stroke_number[index] < 15:
            dp_per_temp = list(df_cf['dp_percent_per_stroke'][index][:(df_cf.stroke_number[index])]) + [-1]*(15-df_cf.stroke_number[index])
            dp_per_temp = percentfierv(np.array(dp_per_temp))
            dirtemp = list(df_cf['direction'][index][:(df_cf.stroke_number[index])]) + [-9999]*(15-df_cf.stroke_number[index])
            tptemp = list(df_cf['time_per_stroke'][index][:(df_cf.stroke_number[index])]) + [-9999]*(15-df_cf.stroke_number[index])
            dptemp = list(df_cf['dp_per_stroke'][index][:(df_cf.stroke_number[index])]) + [-9999]*(15-df_cf.stroke_number[index])
        else:
            dp_per_temp = list(df_cf['dp_percent_per_stroke'][index][:15])
            dp_per_temp = percentfierv(np.array(dp_per_temp))
            dirtemp = list(df_cf['direction'][index][:15])
            tptemp = list(df_cf['time_per_stroke'][index][:15])
            dptemp = list(df_cf['dp_per_stroke'][index][:15])

        for str_n in xrange(15):
            df_cf.loc[index,'datapoint_percentage_stroke{}'.format(str_n)] = dp_per_temp[str_n]
            df_cf.loc[index,'direction_stroke{}'.format(str_n)] = dirtemp[str_n]
            df_cf.loc[index,'time_stroke{}'.format(str_n)] = tptemp[str_n]
            df_cf.loc[index,'datapoint_stroke{}'.format(str_n)] = dptemp[str_n]    

    return df_cf

def _percentfier(data):
    '''
    function:
    used within a function "feature_eng_pt4"
    from input data, it will reduce any data that is less than 1 to 0
    and anything that is less than 0, -9999 (null value)
    
    input:
    data
    
    output:
    return percentage, 0 or -9999
    '''
    
    if data*100 >= 1:
        return data
    elif data < 0:
        return -9999
    else: 
        return 0

def _radian_direction(dy,dx):
    '''
    function:
    used within a function "feature_eng_pt3"
    from input dy and dx, it will determine the direction of that vector.
    
    input:
    dx = delta x (int)
    dy = delta y (int)
    
    output:
    direction (int) (range= 0:6.28)
    '''
    if dy < 0.0 and dx > 0.0:
        return (2*np.pi + np.arctan(dy/dx))
    elif dy >=0.0 and dx > 0.0:
        return (np.arctan(dy/dx))
    else:
        return np.pi + np.arctan(dy/dx)

def feature_eng_pt_test(df_cf):
    dummies = pd.get_dummies(df_cf['countrycode'], prefix='country', drop_first=True)
    return  df_cf.join(dummies)