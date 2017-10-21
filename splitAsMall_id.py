
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd


# In[21]:


data = pd.read_csv('user_shop_behavior_cleanned')
shop = pd.read_csv('shop_info_cleanned')
eva = pd.read_csv('evalution_pulic_cleanned')


# In[3]:


data.head(5)


# In[5]:


eva.head(5)


# In[26]:



malls=pd.DataFrame(shop['mall_id'])
malls=malls[['mall_id']]
malls.drop_duplicates(inplace=True)

malls.count()


# In[27]:


train = data
malls=malls.mall_id.values.tolist()
print('malls type:',type(malls[0]))

#为train训练集添加mall_id
shop=shop[['shop_id','mall_id']]
shop.drop_duplicates(inplace=True)
train=pd.merge(train,shop,on=['shop_id'],how='left')
train.drop_duplicates(inplace=True)


#防止数据集过大，切分数据集


print('split dataset finished')


# In[28]:


for mall in malls:
    train_t=train[train['mall_id']==mall]
    train_t.to_csv(mall+'.csv',index=None)
    
    


# In[ ]:




