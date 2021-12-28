#!/usr/bin/env python
# coding: utf-8

# In[127]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[128]:


#importing csv file
df = pd.read_csv('churn_clean.csv')


# In[129]:


#Getting Dummy Variables
df = pd.get_dummies(df, drop_first = True)


# In[130]:


df.info()


# In[131]:


df1 = df[['Age','Children','Income','Outage_sec_perweek','Phone_Yes', 'MonthlyCharge', 'Churn_Yes']]


# In[132]:


df1


# In[133]:


#prihting our cleaned data to csv
df1.to_csv ('cleaned_data_task2.csv')


# In[134]:


df1.columns


# In[135]:


df1.describe()


# In[136]:


pd.value_counts(df['Churn_Yes'])


# In[137]:


pd.value_counts(df['Phone_Yes'])


# In[42]:


#plotting using pairplot
sns.pairplot(df1)


# In[140]:


#correlation without reg line

sns.jointplot(data = df1, x = df1['MonthlyCharge'], y = df1['Churn_Yes'], kind="reg",
              logx=True,)


# In[59]:


from sklearn.model_selection import train_test_split


# In[64]:


X = df1[['Age','Children','Income','Outage_sec_perweek','Phone_Yes','MonthlyCharge']]
y = df1['Churn_Yes']


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[66]:


from sklearn.linear_model import LogisticRegression


# In[67]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[68]:


predictions = logmodel.predict(X_test)


# In[76]:


from sklearn.metrics import classification_report, confusion_matrix


# In[77]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[72]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[73]:


Xc = sm.add_constant(X)
logistic_regression = sm.Logit(y,Xc)
fitted_model = logistic_regression.fit()


# In[74]:


fitted_model.summary()


# In[119]:


X = df1[['Phone_Yes','MonthlyCharge']]
y = df1['Churn_Yes']


# In[120]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[121]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[122]:


predictions = logmodel.predict(X_test)


# In[123]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[124]:


Xc = sm.add_constant(X)
logistic_regression = sm.Logit(y,Xc)
fitted_model = logistic_regression.fit()
fitted_model.summary()


# In[ ]:




