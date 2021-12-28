#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd


# In[14]:


#cleaning our data
df = pd.read_csv('churn_clean.csv')
df
df_new = df[['Phone','Tablet','Multiple','OnlineSecurity','OnlineBackup','DeviceProtection']]
df_new = pd.get_dummies(df_new, drop_first=True)
df_new.to_csv ('cleaned_data_d212_task3.csv')


# In[15]:


# Import apriori from mlxtend
from mlxtend.frequent_patterns import apriori


# In[16]:


# Compute frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(df_new, min_support = .003, 
                            max_len = 1, 
                            use_colnames = True)
print(frequent_itemsets.head())


# In[17]:


from mlxtend.frequent_patterns import association_rules

# Compute frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(df_new, min_support = 0.3, 
                            max_len = 2, use_colnames = True)

# Compute all association rules for frequent_itemsets
rules = association_rules(frequent_itemsets, 
                            metric = "lift", 
                         min_threshold = 1.0)

# Print association rules
print(rules)


# In[18]:


# Import the association rules function
from mlxtend.frequent_patterns import apriori, association_rules

# Compute frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(df_new, min_support = 0.0015, 
                            max_len = 2, use_colnames = True)

# Compute all association rules using confidence
rules = association_rules(frequent_itemsets, 
                            metric = "confidence", 
                            min_threshold = 0.5)

# Print association rules
print(rules)


# In[19]:


# Apply the apriori algorithm with a minimum support of 0.0001
frequent_itemsets = apriori(df_new, min_support = 0.0001, use_colnames = True)

# Generate the initial set of rules using a minimum support of 0.0001
rules = association_rules(frequent_itemsets, 
                          metric = "support", min_threshold = 0.0001)

# Set minimum antecedent support to 0.35
rules = rules[rules['antecedent support'] > 0.35]

# Set maximum consequent support to 0.35
rules = rules[rules['consequent support'] < 0.35]

# Print the remaining rules
print(rules)


# In[ ]:




