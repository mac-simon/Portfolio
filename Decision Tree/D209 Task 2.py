#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


#cleaning our data
df = pd.read_csv('churn_clean.csv')
df = pd.get_dummies(df)
df = df[['Age','Children','Income','Outage_sec_perweek','Phone_Yes', 'MonthlyCharge', 'Churn_Yes']]


# In[25]:


#printing head of data
df.head()


# In[26]:


#importing our standard scaler package
from sklearn.preprocessing import StandardScaler
#creating our Standard Scaler Model
scaler = StandardScaler()


# In[27]:


#fitting our Standard Scaler Model
scaler.fit(df.drop('Churn_Yes', axis =1))


# In[28]:


#transforming our data
scaled_features = scaler.transform(df.drop('Churn_Yes', axis =1))
#Printing our new scaled data with Churn Yes
cleaned_data = pd.DataFrame(scaled_features, columns = df.columns[1:])
cleaned_data.to_csv ('cleaned_data_d209_task2.csv')
print(cleaned_data)


# In[29]:


#setting up our predictor variables 
df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_feat.head()


# In[30]:


#importing our train test split and setting up our variables
from sklearn.model_selection import train_test_split

X = df_feat
y = df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[31]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[32]:


predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[33]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[16]:


X_train.to_csv('X_training Data_d209_task2')
y_train.to_csv('y_training Data_d209_task2')
X_train.to_csv('X_test Data_d209_task2')
y_train.to_csv('y_test Data_d209_task2')


# In[ ]:




