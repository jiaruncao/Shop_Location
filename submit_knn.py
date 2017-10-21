
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[4]:


data = pd.read_csv('user_shop_behavior_cleanned')
eva = pd.read_csv('AB榜测试集-evaluation_public.csv')


# In[5]:


location = data['longitude']
location = pd.DataFrame(location)
location.insert(column= 'latitude',value=data['latitude'],loc = 0)
location= np.array(location)


# In[6]:


label = pd.DataFrame(data['shop_id'])
label = np.array(label)


# In[7]:


knn = KNeighborsClassifier()
knn.fit(location,label)


# In[8]:


location_test = eva['longitude']
location_test = pd.DataFrame(location_test)
location_test.insert(column= 'latitude',value=eva['latitude'],loc = 0)
location_test = np.array(location_test)


# In[9]:


location_test.shape


# In[34]:


predict = knn.predict(location_test)


# In[17]:





# In[35]:


shop_id =pd.DataFrame(predict,columns=['shop_id'])
result = shop_id
result.insert(column='row_id',value=eva['row_id'],loc = 0)
result


# In[36]:


result.to_csv('result.csv',index = 0)


# In[ ]:




