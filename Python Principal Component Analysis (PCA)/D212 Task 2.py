#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


#cleaning our data
df = pd.read_csv('churn_clean.csv')
df = pd.get_dummies(df)
df = df[['Income','Outage_sec_perweek','MonthlyCharge', 'Bandwidth_GB_Year']]
df.head()


# In[61]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
scaled_data_clean = pd.DataFrame(df_scaled,columns=df.columns[:])

scaled_data_clean.head()
scaled_data_clean.to_csv ('cleaned_data_d212_task2.csv')


# In[62]:


# Create the correlation matrix
corr = scaled_data_clean.corr()

# Draw the heatmap
sns.heatmap(corr,center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()

sns.pairplot(scaled_data_clean)


# In[63]:


# Apply PCA
pca = PCA(n_components=4)
pca.fit(df_scaled)


# In[64]:


print(pca.explained_variance_ratio_)


# In[65]:


print(pca.explained_variance_ratio_.cumsum())


# In[66]:


explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))
pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3','PC4'], columns=['PC'])
explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])
df_explained_variance = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
df_explained_variance


# In[67]:


sns.lineplot(x='PC',y='Explained Variance',data=df_explained_variance).set_title('Scree Plot')


# In[68]:


sns.lineplot(x='PC',y='Cumulative Variance',data=df_explained_variance).set_title('Total Variance')


# In[69]:


# Apply PCA
pca = PCA(n_components=2)
pca.fit(df_scaled)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())


# In[70]:


# Create the correlation matrix
corr = df_explained_variance.corr()

# Draw the heatmap
sns.heatmap(corr,center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()

sns.pairplot(df_explained_variance)


# In[ ]:




