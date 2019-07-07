#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as LE


# In[2]:


train_data = pd.read_csv('/Users/macbook/Desktop/CharanbRamesh_Drive/Neural_Nets/datasets/titanic/train.csv')


# In[3]:


train_data.head()


# In[4]:


pd.isna(train_data).sum()


# In[5]:


train_data = train_data.set_index('PassengerId')


# In[6]:


train_data = train_data.drop(['Cabin','Name','Ticket'],axis = 1)


# In[7]:


train_data['Age'] = train_data['Age'].fillna(value = train_data['Age'].mean())


# In[8]:


train_data = train_data.dropna(axis=0)


# In[9]:


pd.isna(train_data).sum()


# In[10]:


train_data[0:10]


# In[11]:


le = LE()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
train_data['Embarked'] = le.fit_transform(train_data['Embarked'])


# In[12]:


train_data[0:10]


# In[13]:


train_data.describe()


# In[14]:


train_data['Age'] = (tf.keras.utils.normalize(np.array(train_data['Age']),order=2)).reshape(-1,1)


# In[15]:


train_data['Fare'] = (tf.keras.utils.normalize(np.array(train_data['Fare']),order=2)).reshape(-1,1)


# In[16]:


train_data[0:10]


# In[17]:


X = train_data.drop('Survived',axis = 1)


# In[18]:


y = train_data['Survived']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# In[20]:


len(X_train),len(X_test)


# In[21]:


len(y_train), len(y_test)


# In[ ]:




