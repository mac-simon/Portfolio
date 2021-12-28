#!/usr/bin/env python
# coding: utf-8

# In[300]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


# In[301]:


df = pd.read_csv('teleco_time_series .csv', index_col=0)
df = df[1:]
df.head()


# In[302]:


#plotting initial model
df['Revenue'].plot()
plt.title("Revenue over Time")
plt.xlabel("Days")
plt.ylabel("Revenue")


# In[303]:


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Revenue'], model='additive',period=365)  
result.plot();


# In[304]:


rolling_mean = df['Revenue'].rolling(window = 12).mean()
rolling_std = df['Revenue'].rolling(window = 12).std()
plt.plot(df['Revenue'], color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()


# In[305]:


#Frequency 
from scipy import signal
import matplotlib.pyplot as plt
f, Pxx_den = signal.periodogram(df['Revenue'])
plt.semilogy(f, Pxx_den)
plt.ylim([1e-3, 1e4])
plt.title('Spectrum Analysis')
plt.xlabel('Revenue')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


# In[306]:


#using ad fuller to show our model isn't stationary 
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Revenue'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))


# In[307]:


from pmdarima import auto_arima


# In[308]:


auto_arima(df['Revenue'],seasonal=False).summary()


# In[309]:


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
from statsmodels.tsa.statespace.tools import diff
df['d1'] = diff(df['Revenue'],k_diff=2)

# Equivalent to:
# df1['d1'] = df1['Revenue'] - df1['Revenue'].shift(1)

adf_test(df['d1'],'Revenue')


# In[202]:



#using ad fuller to show our model isn't stationary 
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Revenue'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))


# In[310]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
title = 'Autocorrelation: Revenue'
lags = 40
plot_acf(df['Revenue'],title=title,lags=lags);


# In[311]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
title = 'Partialcorrelation: Revenue'
lags = 40
plot_pacf(df['Revenue'],title=title,lags=lags);


# In[312]:


#AR = 1
#I = 1
#MA = 0 or 1

stepwise_fit = auto_arima(df['Revenue'], start_p=0, start_q=0,
                          max_p=2, max_q=2, m=12,
                          seasonal=False,
                          d=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwise_fit.summary()


# In[313]:


len(df)


# In[314]:


train = df.iloc[:500]
test = df.iloc[500:]
train.to_csv('d213 task 1 training data')
test.to_csv('d213 task 1 testing data')


# In[315]:


from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
model = ARIMA(train['Revenue'],order=(1,1,1))
results = model.fit()
results.summary()


# In[316]:


# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(1,1,1) Predictions')


# In[210]:


predictions


# In[317]:


# Plot predictions against known values
# HERE'S A TRICK TO ADD COMMAS TO Y-AXIS TICK VALUES
import matplotlib.ticker as ticker
formatter = ticker.StrMethodFormatter('{x:,.0f}')
title = 'Revenue Over Time'
ylabel='Revenue'
xlabel='Days' 

ax = test['Revenue'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[318]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(test['Revenue'], predictions)
print(f'ARIMA(1,1,1) MSE Error: {error:11.10}')


# In[319]:


test['Revenue'].mean()


# In[320]:


predictions.mean()


# In[321]:


model = ARIMA(df['Revenue'],order=(1,1,1))
results = model.fit()
fcast = results.predict(len(df),len(df)+11,typ='levels').rename('ARIMA(1,1,1) Forecast')


# In[322]:


# Plot predictions against known values
title = 'Revenue over time'
ylabel='Revenue'
xlabel='Days'

ax = df['Revenue'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[ ]:




