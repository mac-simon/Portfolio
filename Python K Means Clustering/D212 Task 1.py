#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


#cleaning our data
df = pd.read_csv('churn_clean.csv')
df = pd.get_dummies(df)
df = df[['Age','Children','Income','Outage_sec_perweek','MonthlyCharge', 'Churn_Yes']]


# In[47]:


#printing head of data
df.head()


# In[48]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Churn_Yes',axis=1))
scaled_features = scaler.transform(df.drop('Churn_Yes', axis =1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
cleaned_data = pd.DataFrame(scaled_features, columns = df.columns[1:])
cleaned_data.to_csv ('cleaned_data_d212_task1.csv')


# In[49]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df_feat)


# In[50]:


kmeans.cluster_centers_


# In[51]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Churn_Yes'],kmeans.labels_))
print(classification_report(df['Churn_Yes'],kmeans.labels_))


# In[ ]:




