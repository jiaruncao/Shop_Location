
# coding: utf-8

# In[3]:


import pandas as pd

train=pd.read_csv('ccf_first_round_user_shop_behavior.csv')#train_ccf_first_round_user_shop_behavior.csv
test=pd.read_csv('evaluation_public.csv')#AB_test_evaluation_public.csv
shop=pd.read_csv('ccf_first_round_shop_info.csv')#ccf_first_round_shop_info.csv

malls=pd.read_csv('mall_test2.csv')#mall_test2.csv
malls=malls.mall_id.values.tolist()
print('malls type:',type(malls[0]))

#为train训练集添加mall_id
shop=shop[['shop_id','mall_id']]
shop.drop_duplicates(inplace=True)
train=pd.merge(train,shop,on=['shop_id'],how='left')
train.drop_duplicates(inplace=True)
train.to_csv('1train.csv',index=None)

#防止数据集过大，切分数据集
for mall in malls:
    train_t=train[train['mall_id']==mall]
    train_t.to_csv('2train_'+mall+'.csv',index=None)
    
    test_t=test[test['mall_id']==mall]
    test_t.to_csv('2test_'+mall+'.csv',index=None)

print('split dataset finished')

for mall in malls:
    ###############extract feature here###############
    train_t=pd.read_csv('2train_'+mall+'.csv')
    test_t=pd.read_csv('2test_'+mall+'.csv')

    #以提取wifi_info中的bssid,signal和wifi_flag为例
    #train
    s=train_t['wifi_infos'].str.split(';').apply(pd.Series,1).stack()
    s.index=s.index.droplevel(-1)
    s.name='wifi_info'
    train_t=train_t.join(s)
    train_t=train_t.reset_index()
    train_t['bssid']=train_t.wifi_info.apply(lambda s:s.split('|')[0])
    train_t['signal']=train_t.wifi_info.apply(lambda s:s.split('|')[1])
    train_t['wifi_flag']=train_t.wifi_info.apply(lambda s:s.split('|')[2])

    train_t.drop('wifi_infos',axis=1,inplace=True)
    train_t.drop('wifi_info',axis=1,inplace=True)

    #test
    s=test_t['wifi_infos'].str.split(';').apply(pd.Series,1).stack()
    s.index=s.index.droplevel(-1)
    s.name='wifi_info'
    test_t=test_t.join(s)
    test_t=test_t.reset_index()
    test_t['bssid']=test_t.wifi_info.apply(lambda s:s.split('|')[0])
    test_t['signal']=test_t.wifi_info.apply(lambda s:s.split('|')[1])
    test_t['wifi_flag']=test_t.wifi_info.apply(lambda s:s.split('|')[2])

    test_t.drop('wifi_infos',axis=1,inplace=True)
    test_t.drop('wifi_info',axis=1,inplace=True)

    #将bssid由string型转化为数值型，bssid样例：b_4162269
    #弃用LabelEncoder方法
    # lbl=preprocessing.LabelEncoder()
    # lbl.fit(list(set(train['bssid'].values)|set(test['bssid'].values)))
    # train['bssid']=lbl.transform(train['bssid'].values)
    # test['bssid']=lbl.transform(test['bssid'].values)
    train_t['bssid']=train_t.bssid.apply(lambda s:s[2:])
    train_t['bssid']=train_t.bssid.astype('int')
    test_t['bssid']=test_t.bssid.apply(lambda s:s[2:])
    test_t['bssid']=test_t.bssid.astype('int')

    #将shop_id由string型转化为数值型，shop_id样例：s_1126
    train_t['shop_id']=train_t.shop_id.apply(lambda s:s[2:])
    train_t['shop_id']=train_t.shop_id.astype('int')

    train_t.reset_index()
    test_t.reset_index()
    train_t.to_csv('3train_'+mall+'.csv',index=None)
    test_t.to_csv('3test_'+mall+'.csv',index=None)
    print('train_t_'+mall+' columns',train_t.columns)
    print('test_t_'+mall+' columns',test.columns)
    #break#用于调试


# In[4]:


print ('hello')


# In[ ]:




