# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:26:02 2017

@author: saurabh.t.singh
"""

conda install statsmodels

import pandas as pd
import sklearn as sk
import numpy as np
from pandasql import *
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score




psql = lambda q: sqldf(q,globals())

######################################
def varfreq(df_model_in,varlist_in, dependent_in):
    fframe = pd.DataFrame()
    print("1")
    for x in varlist_in:    
        ct = pd.crosstab(df_model_in[x],df_model[dependent_in])
        print("2")
        ct['var'] = x
        ct['values'] = ct.index
        print("3")
        ct = ct[ct['values'] == 1]
        print("4")
        ct = pd.melt(ct, id_vars=['var','values'], value_vars=[0,1])
        print("5")
        ct = ct.rename(columns={'attrited':'Response','value':'count'})
        print("6")
        ct['total'] = sum(ct['count'])
        print("7")
        ct['percent'] = ct['count']/ct['total']
        print("8")
        fframe = fframe.append(ct)
    return fframe

######################################

def variabledistribution(dfvd, varlist, dependent):
    print("1")
    fframe = pd.DataFrame()
    print("2")
    for x in varlist:
        print("3")
        print(x)
        tframe = pd.DataFrame()
        dfvd['decile'] = pd.qcut(dfvd[x], 10, labels=False)
        print("4")
        #tframe['Rank'] = df['decile'].unique
        gby = dfvd.groupby(by=dfvd['decile'])
        print("5")
        tframe['count'] = gby[x].count()
        tframe['var'] = x
        tframe['rank'] = tframe.index
        tframe['min'] = gby[x].min()
        tframe['max'] = gby[x].max()
        tframe['avg'] = gby[x].mean()
        
        tframe['count_dep'] = gby[dependent].sum()
        tframe['mean_dep'] = gby[dependent].sum() / gby[dependent].count()
        print("6")
        print(tframe)
        fframe = fframe.append(tframe)
        fframe.set_index(['var','rank'])
        print("7")
        print(fframe)
        dfvd.drop('decile',axis=1)
        
    return fframe

######################################

def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)
######################################


def rankorder(y_pred, y_model):
    data = pd.DataFrame()
    data['response'] = y_model
    data['nonresponse'] = 1 - data['response']
    data['score'] = y_pred
    data['decile'] = pct_rank_qcut(data.score,10)
    grouped = data.groupby('decile', as_index = False)
    agg1 = pd.DataFrame()
    agg1['min_scr'] = grouped.min().score
    agg1['max_scr'] = grouped.max().score
    agg1['response'] = grouped.sum().response
    agg1['total'] = grouped.count().response
    agg1['nonresponse'] = agg1['total'] - agg1['response']
    
    agg2 = (agg1.sort_index(by = 'min_scr',ascending=False)).reset_index(drop = True)
    agg2['odds'] = (agg2.response / agg2.nonresponse).apply('{0:.2f}'.format)
    agg2['response_rate'] = (agg2.response / agg2.total).apply('{0:.2%}'.format) 
    agg2['cumperresponse'] = (agg2.response / data.response.sum()).cumsum()
    agg2['cumpernonresponse'] = (agg2.nonresponse / data.nonresponse.sum()).cumsum()
    agg2['ks'] = np.round((agg2['cumperresponse'] - agg2['cumpernonresponse']), 4) * 100
    flag = lambda x: '<----' if x == agg2.ks.max() else ''
    agg2['max_ks'] = agg2.ks.apply(flag)
    return agg2
######################################


df = pd.read_excel("C:\\UCB\\Import\\model\\Data_for_rf.xlsx")
df.dtypes
df['attrited'] = df['Leaving'].notnull().astype(int)
df['age'] = df['MAL'] + 240

# create varlist ######
out = pd.DataFrame()
out['variable'] = df.columns
out['missing_count'] = [sum(df[x].isnull()) for x in df.columns]
out['count'] = [sum(df[x].notnull()) for x in df.columns]
  
## call variabledistribution ######  
varlist1 = ['In_EUR_17_New']
dependent1 = 'attrited'
f = variabledistribution(df, varlist1, dependent1)

##create dummy variables
df['mal_1'] = np.where(df['MAL'] <= 11, 1, 0 )
df['mal_2'] = np.where(df['MAL'] > 11, 1, 0 )


df['age_1'] = np.where(df['age'] <= 251, 1, 0 )
df['age_2'] = np.where(df['age'] > 251, 1, 0 )

df['Comp_seg_1'] = np.where(df['Comp_seg'] == 1, 1, 0 )
df['Comp_seg_2'] = np.where(df['Comp_seg'] > 1, 1, 0 )

df['Gender_m'] = np.where(df['Gender'] == 'MALE', 1, 0 )

df['Skill_Type'] = ['Super Niche' if (x == 'Super Niche' or x == 'Super niche') else x for x in df['Skill_Type']]
df['Skill_Type'] = ['Regular' if (x == 'Regular-10 weeks' or x == 'regular') else x for x in df['Skill_Type']]
st_dummies = pd.get_dummies(df['Skill_Type'], prefix='skill_type')
df = pd.concat([df,st_dummies],axis=1)

##### model dataset

df_model = df[['Attried','salary_growth_lt_15_0','salary_growth_lt_10_0','salary_growth_lt_1_14','salary_growth_gt_14',
               'Average_growth_0_5',	'Average_growth_gt_5',	'Average_growth_lt_0','Salary_growth_0_3','Salary_growth_gt_3_5',	'Salary_growth_lt_0',	
               'Month_not_promo_0_20','Month_not_promo_lt_30','Month_not_promo_gt_30','age_lt_45',	
               'age_lt_45_55',	'no_promo_0',	'no_promo_1',	'no_promo_23',	'Compa_ratio_above125',	
               'Compa_ratio_below75',	'tor_top_talent',	'tor_top_talent_rise_star',	'tor_effective_per',	
               'tor_core_per',	'tor_trusted_prof',	'tor_low_per',	'tor_not_rated',	'Perform_not_met_16',	
               'Perform_met_16','Perform_overachieved_16',	'Behaviour_2016_inline','Behaviour_2016_notinline',
               'Rehired_1_0',
                'Company_car_avail']]

# create varlist ######
out = pd.DataFrame()
out['variable'] = df_model.columns
out['missing_count'] = [sum(df_model[x].isnull()) for x in df_model.columns]
out['count'] = [sum(df_model[x].notnull()) for x in df_model.columns]


#### varFreq ########
varlist1 = ['salary_growth_lt_15_0','salary_growth_lt_10_0','salary_growth_lt_1_14','salary_growth_gt_14',
               'Average_growth_0_5',	'Average_growth_gt_5',	'Average_growth_lt_0','Salary_growth_0_3','Salary_growth_gt_3_5',	'Salary_growth_lt_0',	
               'Month_not_promo_0_20','Month_not_promo_lt_30','Month_not_promo_gt_30','age_lt_45',	
               'age_lt_45_55',	'no_promo_0',	'no_promo_1',	'no_promo_23',	'Compa_ratio_above125',	
               'Compa_ratio_below75',	'tor_top_talent',	'tor_top_talent_rise_star',	'tor_effective_per',	
               'tor_core_per',	'tor_trusted_prof',	'tor_low_per',	'tor_not_rated',	'Perform_not_met_16',	
               'Perform_met_16','Perform_overachieved_16',	'Behaviour_2016_inline','Behaviour_2016_notinline',
               'Rehired_1_0',
                'Company_car_avail']
dependent1 = 'Attried'

df_varfreq = varfreq(df_model,varlist1, dependent1)
df_varfreq.to_excel('C:\\UCB\\Import\\model\\python\\Variable_Selection.xlsx')

x_model = df_model[['salary_growth_lt_15_0',
'salary_growth_lt_10_0',
'salary_growth_lt_1_14',
'Average_growth_0_5',
'Average_growth_lt_0',
'Salary_growth_gt_3_5',
'Salary_growth_lt_0',
'Month_not_promo_0_20',
'Month_not_promo_lt_30',
'age_lt_45_55',
'no_promo_1',
'no_promo_23',
'Compa_ratio_below75',
'tor_top_talent',
'tor_top_talent_rise_star',
'tor_effective_per',
'tor_trusted_prof',
'tor_low_per',
'tor_not_rated',
'Perform_met_16',
'Perform_overachieved_16',
'Behaviour_2016_inline',
'Rehired_1_0',
'Company_car_avail'
]]
y_model = df_model[['Attried']]

####### bivariate ######
# instantiate our model

model = sm.Logit(y_model,x_model)
res = model.fit(maxiter=1)
res.summary()


####### VIF #########

x_model_ = x_model.drop(['tor_top_talent_rise_star','Perform_overachieved_16','no_promo_1','Month_not_promo_0_20'],1)
features = x_model_.columns.tolist()
#features = df_model_.columns.tolist()
features = features.remove('Month_not_promo_0_20')
features.remove('Attried')

featuresstring = 'Attried~' + "+".join(features)
# Break into left and right hand side; y and X

df_model_ = df_model.drop(['tor_top_talent_rise_star','Perform_overachieved_16','no_promo_1'],1)
y, X = dmatrices(featuresstring, df_model_, return_type="dataframe")

# For each Xi, calculate VIF

vif = pd.DataFrame()
for i in range(X.shape[1]):
    
    vif['var'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif.to_excel('C:\\UCB\\Import\\model\\python\\vif.xlsx')    

######### model ########

#features = x_model.columns.tolist()

x_model = df_model[features]
y_model = df_model[['Attried']]

model = sm.Logit(y_model,x_model)
res = model.fit()
res.summary()
res.aic
res.bic
Parameters = np.exp(res.params)


print("Test Accuracy  :: ", accuracy_score(y_model, np.where(res.predict(x_model) > 0.5,1,0)))
print(" Confusion matrix ", confusion_matrix(y_model, np.where(res.predict(x_model) > 0.5,1,0)))


######### rankorder ########
y_pred = res.predict(x_model)
df_rankorder = rankorder(y_pred, y_model)



#CV
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, x_model, y_model, cv=kfold, scoring=scoring)

***************************



