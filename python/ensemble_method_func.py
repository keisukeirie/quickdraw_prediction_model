from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import pickle
np.random.seed(32113)


def data_preparer_ensemble(df1,df2,df3,df4, lbl = 'word', countries=['US','BR','RU','KR'],\
                   words=['cat','tiger','lion','dog'],sample=30000, limit = 5000):
    '''
    Function:
    process dataframes so that it can be used for xgboost, randomforest other ensemble methods
    function prepares dataframe for image recognition model and country code prediction model
    Input:
    df1,2,3,4 = dataframes with different topics (cat,dog,lion,tiger) [dataframe]
    lbl = "word" or "coountrycode":
        word is used when running image recognition.
        countrycode is used when running countrycode prediction.

    countries = list of string that contains country codes of interest [list]
    words = list of string that contains words of topic of interest [list]
    sample = max number of data to take in (used when lbl = word)
    limit = max number of data from one country (used when lbl = countrycode)

    Output:
    new_df = dataframe or X data for your model
    Y = label for your model

    note: uses random.seed(32113)
    '''
    if lbl == 'word':
        df_test1 = _df_initial_fixer(df1,words[0],sample)
        df_test2 = _df_initial_fixer(df2,words[1],sample)
        df_test3 = _df_initial_fixer(df3,words[2],sample)
        df_test4 = _df_initial_fixer(df4,words[3],sample)
        print len(df_test1),len(df_test2),len(df_test3),len(df_test4)

        new_df = pd.concat([df_test1,df_test2,df_test3,df_test4], axis =0)
        yd = new_df.pop('countrycode')
        Y = new_df.pop('word')
        b_loon={}
        for i in xrange(len(words)):
            b_loon[words[i]] = i
        Y = Y.map(b_loon)
        return new_df,Y

    elif lbl == 'countrycode':
        df_test1 = _df_initial_fixer_cc(df1,words[0])
        df_test2 = _df_initial_fixer_cc(df2,words[1])
        df_test3 = _df_initial_fixer_cc(df3,words[2])
        df_test4 = _df_initial_fixer_cc(df4,words[3])
        print len(df_test1),len(df_test2),len(df_test3),len(df_test4)

        new_df = pd.concat([df_test1,df_test2,df_test3,df_test4], axis =0)
        #filter dataframe with selected countries
        df_cf = new_df[(new_df['countrycode']==countries[0])|(new_df['countrycode']==countries[1])|\
                   (new_df['countrycode']==countries[2])|(new_df['countrycode']==countries[3])]
        print len(df_cf)

        # US
        df_US = _country_initial_fixer(df_cf,countries[0],limit)
        #BR
        df_BR = _country_initial_fixer(df_cf,countries[1],limit)
        #RU
        df_RU = _country_initial_fixer(df_cf,countries[2],limit)
        #KR
        df_KR = _country_initial_fixer(df_cf,countries[3],limit)

        print "number of images for US:{}, BR:{}, RU:{}, KR:{}\n"\
                    .format(len(df_US),len(df_BR),len(df_RU),len(df_KR))

        new_df = pd.concat([df_US,df_BR,df_RU,df_KR], axis=0)
        Y = new_df.pop('countrycode')
        b_loon = {}
        for i in xrange(len(countries)):
            b_loon[countries[i]] = i
        Y = Y.map(b_loon)
        b_loon2={'cat':0,'tiger':1,'lion':2,'dog':3}
        new_df['word']=new_df['word'].map(b_loon2)

        return new_df,Y,df_US,df_BR,df_RU,df_KR
    else:
        print "set your lbl to 'word' or 'countrycode' "

def _df_initial_fixer(df, word, sample=60000):
    '''
    function:
    - prepares training and test X and Y for randomforest, XGboost test
    - ramdomly select rows (image) "sample" times from the df dataframe
    and delete features that are not used in ensemble method modeling

    input:
        df = dataframe. output of 1_feature_engineering_func. [pd.dataframe]
        word = name of topic ig "cat" [str]
        sample = number of sample you want to extract from df [int]

    output:
    new data frame!

    '''
    print "total number of images for df_{}: {}".format(word, len(df))
    random_index = np.random.choice(list(df.index), sample, replace=False)
    df = df.loc[list(random_index)]
    df_test = df.drop(['drawing','key_id','timestamp','recognized','X','Y','time',\
                        'X_per_stroke','Y_per_stroke','time_per_stroke',\
                        'total_time_of_stroke','dp_per_stroke','dp_percent_per_stroke',\
                        'direction'], axis=1)
    return df_test

def _df_initial_fixer_cc(df, word):
    '''
    prepares training and test X and Y for xgboost test for countrycode classifier

    function:
    - prepares training and test X and Y for randomforest, XGboost test
                                                    for countrycode classifier
    - delete features that are not used in ensemble method modeling

    input:
        df = dataframe. output of 1_feature_engineering_func. [pd.dataframe]
        word = name of topic ig "cat" [str]

    output:
    new data frame!

    '''
    df_test = df.drop(['drawing','key_id','timestamp','recognized','X','Y','time',\
                        'X_per_stroke','Y_per_stroke','time_per_stroke',\
                        'total_time_of_stroke','dp_per_stroke','dp_percent_per_stroke',\
                        'direction'], axis=1)
    return df_test

def _country_initial_fixer(df,country,limit):
    '''
    Function:
    extracts data with specific country code and ramdomly select "limit" amount
    of data.

    Input:
    df = dataframe (should contain 'countrycode' features) [dataframe]
    country = should be 2 capital letter country code[string]
    limit = max number of rows (data) you want to take into the new data frame

    Output:
    dataframe contains data from selected country (# of data < limit)

    note: uses random.seed(32113)
    '''
    if df[df['countrycode']==country].count()[0] > limit:
        df_c = df[df['countrycode']==country]
        random_c = np.random.choice(list(df_c.index), limit, replace=False)
        df_c = df_c.loc[list(random_c)]
    else:
        df_c = df[df['countrycode']==country]
    return df_c
