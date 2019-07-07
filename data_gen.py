#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as LE


# In[5]:


def train_data_prep(csv_file):
    train_data = pd.read_csv(os.path.abspath(csv_file))
    train_data = train_data.set_index('PassengerId')
    train_data = train_data.drop(['Cabin','Name','Ticket'],axis = 1)
    train_data['Age'] = train_data['Age'].fillna(value = train_data['Age'].mean())
    train_data['Fare'] = train_data['Age'].fillna(value = train_data['Fare'].mean())
    train_data = train_data.dropna(axis=0)
    le = LE()
    train_data['Sex'] = le.fit_transform(train_data['Sex'])
    train_data['Embarked'] = le.fit_transform(train_data['Embarked'])
    train_data['Age'] = (tf.keras.utils.normalize(np.array(train_data['Age']),order=2)).reshape(-1,1)
    train_data['Fare'] = (tf.keras.utils.normalize(np.array(train_data['Fare']),order=2)).reshape(-1,1)
#     train_data['Pclass'] = (tf.keras.utils.normalize(np.array(train_data['Pclass']),order=2)).reshape(-1,1)
#     train_data['Sex'] = (tf.keras.utils.normalize(np.array(train_data['Sex']),order=2)).reshape(-1,1)
#     train_data['Embarked'] = (tf.keras.utils.normalize(np.array(train_data['Embarked']),order=2)).reshape(-1,1)
#     train_data['Parch'] = (tf.keras.utils.normalize(np.array(train_data['Parch']),order=2)).reshape(-1,1)
    X = train_data.drop('Survived',axis = 1)
    y = train_data['Survived']
    return(train_test_split(X, y, test_size=0.10, random_state=42))


# In[6]:


def test_data_prep(csv_file):
    train_data = pd.read_csv(os.path.abspath(csv_file))
    train_data = train_data.set_index('PassengerId')
    train_data = train_data.drop(['Cabin','Name','Ticket'],axis = 1)
    train_data['Age'] = train_data['Age'].fillna(value = train_data['Age'].mean())
    train_data['Fare'] = train_data['Age'].fillna(value = train_data['Fare'].mean())
    train_data = train_data.dropna(axis=0)
    le = LE()
    train_data['Sex'] = le.fit_transform(train_data['Sex'])
    train_data['Embarked'] = le.fit_transform(train_data['Embarked'])
    train_data['Age'] = (tf.keras.utils.normalize(np.array(train_data['Age']),order=2)).reshape(-1,1)
    train_data['Fare'] = (tf.keras.utils.normalize(np.array(train_data['Fare']),order=2)).reshape(-1,1)
    return(train_data)


# In[ ]:




