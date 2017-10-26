
# coding: utf-8

# In[225]:


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[226]:


malls=pd.read_csv('mall_test2.csv')#mall_test2.csv
malls=malls.mall_id.values.tolist()
#data = pd.read_csv('mySplitData/m_615.csv')


# In[ ]:





# In[228]:


for mall in malls:
    data = pd.read_csv('mySplitData/'+mall+'.csv')
    test = pd.read_csv('splitTestData/'+mall+'.csv')
    wifi_infos1 = test['wifi_infos']

    list_wifi1 = []

    dict_wifi1 = {}

    for wifi_info1 in wifi_infos1:
  #  print wifi_info1
        wifi1 = wifi_info1.split(';')
        for wf1 in wifi1 :
      #  print wf1
            bssid1 = str(wf1.split('|')[0])
            signal1 = float(wf1.split('|')[1])
            dict_wifi1[bssid1]  = signal1
       # print dict_wifi1
            sorted_dict_wifi1 = sorted(dict_wifi1.items(),key=lambda x:x[1],reverse=True)
   #     print sorted_dict_wifi1
        list_wifi1.append(sorted_dict_wifi1)
        dict_wifi1 = {}
        sorted_dict_wifi1={}
    allWifi1 = []

#print list_wifi1
    for mywifi1 in list_wifi1:
        for wifi1 in mywifi1:
            allWifi1.append(wifi1[0])
#print len(allWifi1)
    allWifi1 = list(set(allWifi1))



#--------------------------------------------------------

    wifi_infos = data['wifi_infos']
    list_wifi = []
    dict_wifi = {}
    count = 0
    for wifi_info in wifi_infos:
        wifi = wifi_info.split(';')
        for wf in wifi :
            bssid = str(wf.split('|')[0])
            signal = float(wf.split('|')[1])
            dict_wifi[bssid]  = signal
            sorted_dict_wifi = sorted(dict_wifi.items(),key=lambda x:x[1],reverse=True)
        list_wifi.append(sorted_dict_wifi)
        dict_wifi = {}
        sorted_dict_wifi={}
    allWifi = []
    for mywifi in list_wifi:
        for wifi in mywifi:
            allWifi.append(wifi[0])
   # print len(allWifi)
    allWifi = list(set(allWifi).union(set(allWifi1)))
  #  print len(allWifi)
    allWifi_sorted = []
    for singleWifi in allWifi:
        singleWifi = str(singleWifi).strip('b')
        singleWifi = str(singleWifi).strip('_')
        singleWifi = int(singleWifi)
        allWifi_sorted.append(singleWifi)
    allWifi_sorted = sorted(allWifi_sorted)
    Wifi_sorted = []
    for i in allWifi_sorted:
        i_ = 'b_' + str(i)
        Wifi_sorted.append(i_)
    wifi_matrix  = np.zeros([int(data['wifi_infos'].count()),len(allWifi_sorted)])
    counter = 0
    for wifi in list_wifi:
        for wf in wifi:
            index_wifi = Wifi_sorted.index(str(wf[0]))
            wifi_matrix[counter,index_wifi] = int(wf[1])
        counter += 1

    
    wifi_matrix1  = np.zeros([int(test['row_id'].count()),len(allWifi_sorted)])
    counter1 = 0
    for wifi1 in list_wifi1:
        for wf1 in wifi1:
     #   print wf[1]
            index_wifi1 = Wifi_sorted.index(str(wf1[0]))
    #    print index_wifi
            wifi_matrix1[counter1,index_wifi1] = int(wf1[1])
        counter1 += 1
    label = list(data['shop_id'])
    knn = KNeighborsClassifier(n_jobs= -1,algorithm='kd_tree')
    knn.fit(wifi_matrix,label)
    predict = knn.predict(wifi_matrix1)
    predict = pd.DataFrame(predict,columns=['shop_id'])
    pre_result = pd.concat([test['row_id'],predict['shop_id']],axis = 1)
    result = pd.DataFrame(columns=['row_id','shop_id'])
    result = pd.concat([result,pre_result],axis = 0)
    print mall
result.to_csv('result.csv',index = 0)


# In[138]:


list_wifi[1]


# In[121]:


data['wifi_infos'].count()


# In[124]:


allWifi = []
for mywifi in list_wifi:
    for wifi in mywifi:
        allWifi.append(wifi[0])
len(allWifi)


# In[126]:


allWifi = list(set(allWifi))
allWifi_sorted = []
len(allWifi)


# In[127]:


for singleWifi in allWifi:
    singleWifi = str(singleWifi).strip('b')
   # print singleWifi
    singleWifi = str(singleWifi).strip('_')
   # print singleWifi
    singleWifi = int(singleWifi)
    allWifi_sorted.append(singleWifi)


# In[161]:


allWifi_sorted = sorted(allWifi_sorted)
Wifi_sorted = []


# In[162]:


for i in allWifi_sorted:
    i_ = 'b_' + str(i)
    Wifi_sorted.append(i_)


# In[167]:


Wifi_sorted[620]


# In[187]:


wifi_matrix  = np.zeros([11491,3067])
counter = 0


# In[188]:


for wifi in list_wifi:
    for wf in wifi:
       # print wf[1]
        index_wifi = Wifi_sorted.index(str(wf[0]))
       # print index_wifi
        wifi_matrix[counter,index_wifi] = int(wf[1])
    counter += 1
    


# In[182]:


list_wifi[0]


# In[189]:


np.count_nonzero(wifi_matrix)


# In[191]:


label = list(data['shop_id'])


# In[195]:


wifi_matrix.shape


# In[196]:


len(label)


# In[197]:


knn = KNeighborsClassifier()


# In[198]:


knn.fit(wifi_matrix,label)


# In[ ]:




