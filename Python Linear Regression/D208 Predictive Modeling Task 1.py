#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


#importing csv file
df = pd.read_csv('churn_clean.csv')


# In[34]:


#cleaning the dataset
df = df[['Age','Income','Children','Outage_sec_perweek', 'Techie', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Phone', 'Multiple', 'StreamingTV','Tenure']]
df = pd.get_dummies(df)
df1 = df.drop(['Techie_No','Phone_No','Multiple_No','StreamingTV_No'], axis = 1)


# In[35]:


#prihting our cleaned data to csv
df1.to_csv ('cleaned_data.csv')


# In[36]:


#final columns selection
df1.columns


# In[37]:


df1.info()


# In[38]:


df1.describe()


# In[39]:


df['Techie_Yes'].value_counts()


# In[40]:


df['Phone_Yes'].value_counts()


# In[41]:


df['Multiple_Yes'].value_counts()


# In[42]:


df['StreamingTV_Yes'].value_counts()


# In[43]:


#plotting using pairplot
sns.pairplot(df1)


# In[44]:


#correlation without reg line

sns.jointplot(data = df1, x = df1['Bandwidth_GB_Year'], y = df1['Tenure'])


# In[45]:


#correlation with reg line

sns.regplot(data = df1, x = df1['Bandwidth_GB_Year'], y = df1['Tenure'])


# In[46]:


#setting up regression analysis variables

y = df1['Tenure']
X = df1[['Age', 'Income', 'Children', 'Outage_sec_perweek', 'MonthlyCharge',
       'Bandwidth_GB_Year','Techie_Yes', 'Phone_Yes',
       'Multiple_Yes', 'StreamingTV_Yes']]


# In[47]:


#importing regression training

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[48]:


#importing linear regression and fitting to our training data

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[49]:


#printing our regression coefficients 
print(lm.coef_)


# In[50]:


predictions = lm.predict( X_test)


# In[51]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[52]:


#graphing our mean of absolute errors 

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[53]:


#graphing residuals 

sns.displot((y_test-predictions),bins=50);


# In[54]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[55]:


import statsmodels.api as sm
results = sm.OLS(y, X).fit()


# In[56]:


print(results.summary())


# In[57]:


#reduced model
y = df1['Tenure']
X = df1[['Age', 'Children', 'MonthlyCharge','Bandwidth_GB_Year','Multiple_Yes', 'StreamingTV_Yes']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
sns.displot((y_test-predictions),bins=50);


# In[58]:


results = sm.OLS(y, X).fit()
print(results.summary())


# In[ ]:




