import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math
warsaw=pd.read_csv('weather prediction/warsaw.csv',parse_dates=['DATE'],index_col='DATE')
warsaw=warsaw[['PRCP','TAVG','TMAX','TMIN','SNWD']]
#I can see the SNWD has a lot of missing data and its not very useful so lets just drop this column

warsaw.drop('SNWD',inplace=True,axis=1)

#lets plot the data to check there are no outliers
plt.plot(warsaw['PRCP'])
plt.title('Precipitation')
sns.despine()
plt.show()

plt.plot(warsaw['TAVG'])
plt.title('Average Temp')
sns.despine()
plt.show()

plt.plot(warsaw['TMIN'])
plt.title('Min Temp')
sns.despine()
plt.show()

plt.plot(warsaw['TMAX'])
plt.title('Max Temp')
sns.despine()
plt.show()

#everything checks out except the Precipitation that shows to have values in the 200's, lets drop the colums with values that are over 30 
warsaw.drop(warsaw[warsaw['PRCP']>30].index,inplace=True)

#to fill data in the min and the max temperature lets fill with the average of the difference with the mean
warsaw['max_diff']=warsaw['TMAX']-warsaw['TAVG']
max_mean=warsaw['max_diff'].mean()

warsaw['min_diff']=warsaw['TAVG']-warsaw['TMIN']
min_mean=warsaw['min_diff'].mean()

warsaw['TMAX'].fillna(warsaw['TAVG']+max_mean,inplace=True)
warsaw['TMIN'].fillna(warsaw['TAVG']-min_mean,inplace=True)

#lets drop the columns that we created
warsaw.drop(['max_diff','min_diff'],axis=1,inplace=True)

#for the Precipitation only 10000 values are missing so lets fill with the average
prcp_avg=warsaw['PRCP'].mean()
warsaw['PRCP'].fillna(prcp_avg,inplace=True)

print(warsaw.info())
#now we don't have any missing data

#lets also get the average of each column by year and plot the data
warsaw['year']=warsaw.index.year
warsaw_pivot=pd.pivot_table(warsaw,index='year').reset_index()
warsaw.drop('year',axis=1,inplace=True)

plt.plot(warsaw_pivot['PRCP'])
plt.title('Precipitation')
sns.despine()
plt.show()

plt.plot(warsaw_pivot['TAVG'])
plt.title('Average Temp')
sns.despine()
plt.show()

plt.plot(warsaw_pivot['TMIN'])
plt.title('Min Temp')
sns.despine()
plt.show()

plt.plot(warsaw_pivot['TMAX'])
plt.title('Max Temp')
sns.despine()
plt.show()

#very intersting it seems that both the average the min and the max temperatures are rising over the years

"""now that we explored the data it is time to predict the future!
lets use the ARIMA model to try and predict the weather.
We'll use the entire MaxTemp colunm as the training model and the last item as the test"""

#first our target column
warsaw['target']=warsaw.shift(-1)['TMAX']
warsaw=warsaw.iloc[:-1,:].copy()
print(warsaw)

train_df=warsaw[:int(len(warsaw)*0.97)]
test_df=warsaw.iloc[int(len(warsaw)*0.97):]

#lets use the ridge model
from sklearn.linear_model import Ridge
reg=Ridge(alpha=0.1)
predictors=['PRCP','TAVG','TMAX','TMIN']

#now lets fit our model
reg.fit(train_df[predictors],train_df['target'])

#now for the predictions
prediction=reg.predict(test_df[predictors])
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(test_df['target'],prediction))#2.17, so on average we were only 2.17 off the true temp, which is not bad at all for a first attempt

#lets see how our model did overall
comb=pd.concat([test_df['target'],pd.Series(prediction,index=test_df.index)],axis=1)
comb.columns=['actual','prediction']
print(comb)#preatty good for a first try

#now lets plot our data to see what our predictions look like
comb.plot()
plt.title('first testing of the model')
plt.show()#the graphs acually look very similer except in the highes and the lows, that means that we did a preatty good job for a first attempt

#lets build a function that will save us to write up this code every time we want to predict the weather

def prediction(predictors,warsaw,reg):
    train_df=warsaw[:int(len(warsaw)*0.97)]
    test_df=warsaw.iloc[int(len(warsaw)*0.97):]
    reg.fit(train_df[predictors],train_df['target'])
    prediction=reg.predict(test_df[predictors])
    error=mean_absolute_error(test_df['target'],prediction)
    comb=pd.concat([test_df['target'],pd.Series(prediction,index=test_df.index)],axis=1)
    comb.columns=['actual','prediction'] 
    return error,comb   


#now to improve the accuracy of our prediction lets add a month rolling mean and some ratios
warsaw['week_max']=warsaw['TMAX'].rolling(7).mean()
warsaw['month_max']=warsaw['TMAX'].rolling(30).mean()
warsaw=warsaw[29:]

#lets remove the days that TMAX=0 and TMIN=0 so avoid infinity
warsaw=warsaw[warsaw['TMAX']!=0]
warsaw=warsaw[warsaw['TMIN']!=0]

warsaw['week_day_max']=warsaw['week_max']/warsaw['TMAX']
warsaw['max_min']=warsaw['TMAX']/warsaw['TMIN']


#now lets test our model again
error,comb=prediction(['PRCP','TAVG','TMAX','TMIN','week_max','week_day_max','max_min'],warsaw,reg)
print(error)#2.13 so a bit of improvment
print(comb)

#lets plot the results
comb.plot()
plt.title('another test with some more statistics')
plt.show()

#lets add some more devitions
warsaw['month_day_average']=warsaw['month_max']/warsaw['TMAX']
#now lets test our model one last time

error,comb=prediction(['PRCP','TAVG','TMAX','TMIN','week_max','week_day_max','max_min','month_day_average'],warsaw,reg)
print(error)#2.13 again, it didnt really improve the result, i guess that we are closing on the limits of this model with this data set
print(comb)

#at last lest look at some statistics
print(reg.coef_)
print(warsaw.corr()['target'])

"""From those statistics we can infer that the colunms that are used the most are 
Average temp and week_max
we also printed the correlations to see if the model is actually working on relevent data
It seems that it is and most of the columns are higely correlated with the data"""

#thats it for now