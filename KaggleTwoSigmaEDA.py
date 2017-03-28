# -*- coding: utf-8 -*-
"""
This script is to do exploratory data analysis for Kaggle competition - Two Sigma Connect
"""
import json
import pandas as pd


""" Read JSON File """
train_df = pd.read_json(r'E:\Kaggle_Two Sigma\Data\train.json')
test_df = pd.read_json(r'E:\Kaggle_Two Sigma\Data\test.json')

"=============================================== "
"Exploratory Data Analysis"
"=============================================== "
""" Data Shape """
print "Training Set: {}".format(train_df.shape)
print "Test Set: {}".format(test_df.shape)

""" Distribution of Dependent Variable in Training Set """
import seaborn as sns
order = ['low','medium','high']
d_var = 'interest_level' #dependent variable
%matplotlib inline
sns.countplot(x=d_var,data=train_df,order=order)

# Distribution of bedrooms by interest_level
# Observation: if number of bedrooms are greater than 4, it tends to be low quality listing
sns.stripplot(x=d_var,y='bedrooms',data=train_df,order=order,jitter=True)
# Distribution of bathrooms by interest_level
# Observation: if number of bathrooms are greater than 4, it tends to be low quality listing
sns.stripplot(x=d_var,y='bathrooms',data=train_df,order=order,jitter=True)
# Mean of bathrooms by category
# Observation: number of bathrooms have slight negative 'correlation' with listing quality
sns.barplot(x=d_var,y='bathrooms',data=train_df,order=order)
# Distribution of price by interest_level
# Observation: there are several outliers higher than 1,000,000. It could be due to input error
# which leads to low quality listing 
sns.stripplot(x=d_var,y='price',data=train_df,order=order,jitter=True)
# Mean of price by interest_level
# It seems price is negative 'correlated' with listing quality. 
# But it might be caused by outliers in price. Will plot median to see if pattern still exists
sns.barplot(x=d_var,y='price',data=train_df,order=order)
import numpy as np
sns.barplot(x=d_var,y='price',data=train_df,order=order,estimator=np.median)
# Observation: seems the pattern still exist

# Now, I am interested to explore the effect of 'features' on listing quality.
# Firstly, I want to see if number of features have an impact on a listing's quality.
num_features = []
for features in train_df['features']:
    num_features.append(len(features))
train_df['num_features'] = num_features
sns.stripplot(x=d_var,y='num_features',data=train_df,order=order,jitter=True)
sns.barplot(x=d_var,y='num_features',data=train_df,order=order)
# Observation: seems like there is no apparent pattern
# Secondly, I want to see if existence of features has an impact

# define a function to create the binary features column
def bool_feature(row):
    if row['num_features'] > 0:
        features = 'yes'
    else:
        features = 'no'
    return features

train_df['bool_features'] = train_df.apply(bool_feature,axis=1)
sns.countplot(x='bool_features',hue=d_var,data=train_df)

# it seems like there is no apparent pattern as well.

# Thirdly, let me choose a random number as a cutoff for number of features to see whether
# that have predictive power. Let us say 12
def bool_feature(row):
    if row['num_features'] > 12:
        features = 'yes'
    else:
        features = 'no'
    return features

train_df['bool_features'] = train_df.apply(bool_feature,axis=1)
sns.countplot(x='bool_features',hue=d_var,data=train_df)
# It seems like still no obvious pattern

# I didn't find an effective feature engineering for variable 'features' at this point.
# Now, let me try some feature engineering for variable 'created' which is the date and time
# a listing is created
# convert unicode to datetime.datetime
train_df['post_datetime'] = train_df['created'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
train_df['post_day'] = train_df['post_datetime'].apply(lambda x: x.day)
def cate_day(x):
    if x<=10:
        period ='beginning'
    elif x<=20:
        period ='middle'
    else:
        period='ending'
    return period

train_df['post_period'] =train_df['post_day'].apply(lambda x:cate_day(x))
sns.countplot(x='post_period',hue=d_var,data=train_df,order=['beginning','middle','ending'])
# Observation: from plot above, we see posting at 1-10th, 11-20th or 21-31th day of a month
# have no impact on interest_level as well

# I will see if weekday/weekend or workday have an impact.
train_df['post_weekday']=train_df['post_datetime'].apply(lambda x:x.weekday()+1)
sns.countplot(x='post_weekday',hue=d_var,data=train_df)

def weekday(x):
    if x==6 or x==7:
        day = "weekend"
    else:
        day = "weekday"
    return day

train_df['post_weekend']=train_df['post_weekday'].apply(lambda x:weekday(x))
sns.countplot(x='post_weekend',hue=d_var,order=['weekday','weekend'],data=train_df)
# Unfortunately, I found no pattern again.
# I want to try if there is pattern in manager_id

'''
before we look for pattern in manager_id, we need to check whether there are common manager_id
between training and testing data. If so, how many are there.
'''
u_manager_train = train_df.manager_id.drop_duplicates()
u_manager_test = test_df.manager_id.drop_duplicates()
print "unique manager id in training set is:{}".format(len(u_manager_train))
print "unique manager id in testing set is:{}".format(len(u_manager_test))
print "number of shared manager id is:{}".format(sum(u_manager_test.isin(u_manager_train)))

train_df_mgid = train_df[['manager_id','interest_level']]
train_df_mgid['count']=1
''' group by manager_id and interest level'''
train_mgid_interest=train_df_mgid.groupby(['manager_id','interest_level'])['count'].sum().reset_index()
''' gorup by manager_id and sorted by count to find most frequent manager_id '''
train_mgid = train_df_mgid.groupby(['manager_id'])['count'].sum().reset_index() 
train_mgid_sorted=train_mgid.sort(columns='count',ascending=False)
''' plot histogram of manager_id '''
sns.barplot(x='manager_id',y='count',data=train_mgid_sorted.iloc[0:100])
''' extract most frequent N manager_id '''
manager_id = train_mgid_sorted.iloc[0:30].manager_id
train_mgid_interest_freq100 = train_mgid_interest.loc[train_mgid_interest['manager_id'].isin(manager_id)]
%matplotlib auto
# use factorplot in seaborn to investigate predictive power of manager_id
sns.factorplot(x='manager_id',y='count',hue='interest_level',data=train_mgid_interest_freq100,kind='bar')
# Observation: great news that some manager consistently post low interest posting. manager_id would 
# be a predicitve variable