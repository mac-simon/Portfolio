#!/usr/bin/env python
# coding: utf-8

# In[115]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[116]:


#cleaning our data
df = pd.read_csv('churn_clean.csv')
df = pd.get_dummies(df)
df = df[['Age','Children','Income','Outage_sec_perweek','Phone_Yes', 'MonthlyCharge', 'Churn_Yes']]


# In[117]:


df.head()


# In[118]:


#importing our standard scaler package
from sklearn.preprocessing import StandardScaler
#creating our Standard Scaler Model
scaler = StandardScaler()


# In[119]:


#fitting our Standard Scaler Model
scaler.fit(df.drop('Churn_Yes', axis =1))


# In[120]:


#transforming our data
scaled_features = scaler.transform(df.drop('Churn_Yes', axis =1))
#Printing our new scaled data with Churn Yes
cleaned_data = pd.DataFrame(scaled_features, columns = df.columns[1:])
cleaned_data.to_csv ('cleaned_data_d209_task1.csv')
print(cleaned_data)


# In[121]:


#setting up our predictor variables 
df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_feat.head()


# In[122]:


from sklearn.model_selection import train_test_split

X = df_feat
y = df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[123]:


print(X_train,y_train)


# In[124]:


print(X_test,y_test)


# In[125]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# In[126]:


pred = knn.predict(X_test)


# In[127]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))


# In[128]:


print(classification_report(y_test,pred))


# In[129]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[130]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[131]:


neighbors = np.arange(1, 20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
    

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[132]:


# NOW WITH K=30

knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[133]:


X_train.to_csv('X_training Data_d209_task1')
y_train.to_csv('y_training Data_d209_task1')
X_train.to_csv('X_test Data_d209_task1')
y_train.to_csv('y_test Data_d209_task1')


# In[ ]:




