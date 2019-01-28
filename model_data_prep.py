# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:43:51 2018

@author: kamalesh.pradhan
"""

import pandas as pd
import numpy as np

# Subsetting for tenure 2-5 yrs and Ps_grade 'Excempt' a
model_data = data_aftr_promo_1[~data_aftr_promo_1['ps_Grade'].isin(['Missing','Non Exempt'])]
model_data = model_data[(model_data['Seniority Grouping']=='2-5Y') & (model_data['Model_data_date_flag']==1) & (pd.isnull(model_data['Username'])==False)]


model_data.to_excel(path + 'model_data.xlsx')

#Female
model_data['Female'] =  np.where(model_data['Gender Key']=='Female',1,0)
#Promoted or not
model_data['Promoted'] =  np.where(model_data['No. of Promotions']>0,1,0)

#
def start_end(row):
    if row['Last Hire Date'].date() < datetime.datetime(2013,1,1).date():
        return row['Last Hire Date']
    else:
        return datetime.datetime(2013,1,1).date()

model_data['Start_date'] = model_data.apply(start_end,axis=1)
def end_end(row):
    if row['Source']=='Demo' :
        return datetime.datetime(2017,11,30).date()
    else:
        return row['Leaving']
model_data['End_date'] = model_data.apply(end_end,axis=1)

# Years
model_data['months_start_to_end'] = (model_data['End_date'].dt.to_period('D') - model_data['Start_date'].dt.to_period('D'))/30
# Average Promotions per month
model_data['years_start_to_end'] = (model_data['months_start_to_end']/12)

# No.of promo per year
model_data['Avg_promotion_per_month'] = (model_data['No_of_promotions_new']/model_data['years_start_to_end'])
# No.of moves per year
model_data['Avg_moves_per_month'] = (model_data['No. of Moves']/model_data['years_start_to_end'])
# Rehired Flag
model_data['Rehired_1_0'] = np.where(pd.isnull(model_data['Rehiring'])==False,1,0)

#Job Family Group
model_data['Job_family_2_5']=["BDevelopment" if x == 'Business Development' else 
                 "PVMP" if  x in(['PVP Marketing & Market Access','PVP Marketing & Patient Access'])  else 
                 "PVFQPSafety" if  x in(['PVF Legal & IP & Ethics & Compliance','PVF QA & Patient Safety'])  else
                 "CSBDev" if  x in(['Corporate Strategy&Business Dev.'])  else
                 "QAPSafety" if  x in(['QA Patient Safety & Risk Mgmt'])  else
                 "Patient_value" if  pd.isnull(x)==False   else
                 "Missing"
                for x in model_data['Text Org Unit N-1(Dynamic)']]

#Company Car Flag
model_data['Company_car_avail'] = np.where(model_data['Company Car 17']=='Yes',1,0)

# Average salary Growth YOY
model_data['Increment_15_to_16'] =   (model_data['In EUR 16'] - model_data['In EUR 15'])/model_data['In EUR 15']  
model_data['Increment_16_to_17'] =   (model_data['In EUR 17 OLD'] - model_data['In EUR 16'])/model_data['In EUR 16']  

model_data['Average_growth_15_to_17'] = (model_data['Increment_16_to_17'] + model_data['Increment_15_to_16'] )/2

Mean_Salary_growth= pd.pivot_table(model_data[model_data['Average_growth_15_to_17']<500], values=['Average_growth_15_to_17'], columns=['Country','ps_Grade'], aggfunc=[np.mean]).reset_index()
Mean_Salary_growth.to_excel(path + "Mean_Salary_growth.xlsx")
#Merging 2017 Mean salary to get salary level below avg or morethan avg

model_data = pd.merge(model_data, Mean_Salary_growth[['Country','ps_Grade','mean']], how='left', left_on=['Country','ps_Grade'], right_on = ['Country','ps_Grade'])
model_data = model_data.rename(columns={'mean':'Mean_salary_growth'})

# Slaary_growth_var_against_mean
model_data['Slaary_growth_var_against_mean'] = (model_data['Average_growth_15_to_17'] - model_data['Mean_salary_growth'])

# Month ndifference betwn 'last_day' and 'Last_promotion_date'
def last_day(row):
    if row['Source'] == 'Terminated':
        return row['Leaving']
    else:
        return datetime.datetime(2017,11,30).date()
model_data['last_day'] = model_data.apply(last_day,axis=1)

# Month Since last promotion
model_data['Month_not_promoted'] = (model_data['last_day'].dt.to_period('D') - model_data['Last_promotion_date_mod'].dt.to_period('D'))/30

# Slaary  var against mean
 Mean_Salary= pd.pivot_table(model_data, values=['In_EUR_17_New'], columns=['Country','ps_Grade'], aggfunc=[np.mean]).reset_index()
 Mean_Salary.to_excel(path + "Mean_Salary.xlsx")
#Merging 2017 Mean salary to get salary level below avg or morethan avg
model_data = pd.merge(model_data, Mean_Salary[['Country','ps_Grade','mean']], how='left', left_on=['Country','ps_Grade'], right_on = ['Country','ps_Grade'])
model_data = model_data.rename(columns={'mean':'2017_EUR_2017_mean'})

model_data['salary_var_against_mean'] = (model_data['In_EUR_17_New'] - model_data['2017_EUR_2017_mean'])/model_data['2017_EUR_2017_mean']

Termination Initiation (VO/IV)
model_data_old = model_data
model_data = model_data_old[~(model_data_old['Termination Initiation (VO/IV)']=='Involuntary')]

#Deciling for salary_var_against_mean
model_data=model_data.sort_values(['salary_var_against_mean'],ascending=[True])
model_data['salary_var_against_mean_decile'] = (pd.qcut(model_data['salary_var_against_mean'], 10, labels=False)) + 1

#Deciling for Slaary_growth_var_against_mean
model_data=model_data.sort_values(['Slaary_growth_var_against_mean'],ascending=[True])
model_data['Salary_growth_var_against_mean_decile'] = (pd.qcut(model_data['Slaary_growth_var_against_mean'], 10, labels=False)) + 1

#Deciling for Slaary_growth_var_against_mean
model_data['Avg_promotion_per_month'] = model_data['Avg_promotion_per_month'].astype(float)
model_data = model_data.sort_values(['Avg_promotion_per_month'],ascending=[True])
model_data['Avg_promotion_per_month_rank'] = model_data['Avg_promotion_per_month'].rank(method='first')
model_data['Avg_promotion_per_month_decile'] = pd.qcut(model_data['Avg_promotion_per_month_rank'].values, 10).codes

model_data.to_pickle( path + 'model_data_05012017')

#Deciling for Slaary_growth_var_against_mean
model_data=model_data.sort_values(['Average_growth_15_to_17'],ascending=[True])
model_data['Average_growth_15_to_17_decile'] = (pd.qcut(model_data['Average_growth_15_to_17'], 10, labels=False)) + 1

#Deciling for Month_not_promoted
model_data['Month_not_promoted'] = model_data['Month_not_promoted'].astype(float)

model_data_for_promo = model_data[model_data['No_of_promotions_new']>0]
model_data_for_promo = model_data_for_promo.sort_values(['Month_not_promoted'],ascending=[True])
model_data_for_promo['Month_not_promoted_rank'] = model_data_for_promo['Month_not_promoted'].rank(method='first')
model_data_for_promo['Month_not_promoted_decile'] = pd.qcut(model_data_for_promo['Month_not_promoted_rank'].values, 10).codes

#Deciling for Slaary_growth_var_against_mean
model_data=model_data.sort_values(['In_EUR_17_New'],ascending=[True])
model_data['In_EUR_17_New_decile'] = (pd.qcut(model_data['In_EUR_17_New'], 10, labels=False)) + 1

model_data['Avg_moves_per_month_1'] = np.where(pd.isnull(model_data['Avg_moves_per_month'])==True,0,model_data['Avg_moves_per_month'] )
#Deciling for moves
model_data['Avg_moves_per_month_1'] = model_data['Avg_moves_per_month_1'].astype(float)
model_data=model_data.sort_values(['Avg_moves_per_month_1'],ascending=[True])
model_data['Avg_moves_per_month_rank'] = model_data['Avg_moves_per_month_1'].rank(method='first')
model_data['Avg_moves_per_month_decile'] = pd.qcut(model_data['Avg_moves_per_month_rank'].values, 10).codes

          
#  Hypothesis Distributions
#salary_var_against_mean_decile
#Average_growth_15_to_17_decile
#Avg_promotion_per_month_decile
#Salary_growth_var_against_mean_decile
#Month_not_promoted_decile
#In_EUR_17_New_decile
#Avg_moves_per_month_decile
index_c = ['Avg_moves_per_month_decile','dummy']
columns_c = ['Source']
values_c = 'Corporate Account'
values_hr = ['Avg_moves_per_month_1'] 

Group_decile= pd.pivot_table(model_data, values=values_hr, columns=index_c, aggfunc=[np.min,np.max]).reset_index()
Group_decile = Group_decile.rename(columns={'amin':'Min','amax':'Max'})

data_Pivot = pd.pivot_table(model_data, index=index_c, columns=columns_c, values=values_c,aggfunc=lambda x: len(x.unique())).reset_index()
data_Pivot['Headcount'] = data_Pivot['Demo'].add(data_Pivot['Terminated'],fill_value=0)
data_Pivot['Attr%'] = data_Pivot['Terminated']/data_Pivot['Headcount']

data_Pivot = pd.merge(data_Pivot, Group_decile, how='left', left_on=index_c, right_on = index_c)


#Chk

model_data_for_promo = model_data[model_data['No_of_promotions_new']>0]
chk = chk.sort_values(['Month_not_promoted'],ascending=[True])
chk['Month_not_promoted_rank'] = chk['Month_not_promoted'].rank(method='first')
chk['Month_not_promoted_decile'] = pd.qcut(chk['Month_not_promoted_rank'].values, 10).codes


#For Character variables
#Female
#age_grp
#Promoted
#Rehired_1_0
#No_of_promotions_new
#Compa ratio grouping 17
#Job_family_2_5
#Company_car_avail
#TOR Rating 2016
#Performance Achievement 2016
#Behaviour 2016
#TOR Rating 2016 New

model_data_subset_nonmiss_beh = model_data[pd.isnull(model_data['Behaviour 2016'])==False]

index_c = ['No_of_promotions_new','dummy']
columns_c = ['Source']
values_c = 'Corporate Account'
data_Pivot = pd.pivot_table(model_data_subset_nonmiss_beh, index=index_c, columns=columns_c, values=values_c,aggfunc=lambda x: len(x.unique())).reset_index()
data_Pivot['Headcount'] = data_Pivot['Demo'].add(data_Pivot['Terminated'],fill_value=0)
data_Pivot['Attr%'] = data_Pivot['Terminated']/data_Pivot['Headcount']

data_promo_gt_3 = data_aftr_promo_1[data_aftr_promo_1['No. of Promotions']==3]

model_data = pd.merge(model_data, model_data_for_promo[['Corporate Account','Month_not_promoted_decile']], how='left', left_on=['Corporate Account'], right_on = ['Corporate Account'])

model_data.to_excel(path + 'model_data_jan052017.xlsx')
model_data.to_pickle(path + 'model_data_jan122017')









start = time.time()
run
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
