
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[26]:


malls=pd.read_csv('mall_test2.csv')#mall_test2.csv
malls=malls.mall_id.values.tolist()
#data = pd.read_csv('mySplitData/m_615.csv')


# In[27]:



#malls = ['m_2058','m_1831']


# In[28]:


for mall in malls:
    data = pd.read_csv('mySplitData/'+mall+'.csv')
    test = pd.read_csv('splitTestData/'+mall+'.csv')
    wifi_infos1 = test['wifi_infos']
    data_loc = pd.concat([data['longitude'],data['latitude']],axis = 1)
    test_loc =  pd.concat([test['longitude'],test['latitude']],axis = 1)
    
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
 #   print allWifi_sorted
    
    Wifi_sorted = []
    for i in allWifi_sorted:
        i_ = 'b_' + str(i)
        Wifi_sorted.append(i_)
#    print Wifi_sorted.index('b_2410')
    wifi_matrix  = np.zeros([int(data['wifi_infos'].count()),len(allWifi_sorted)+2])
    wifi_matrix[:,0:2] = data_loc
    counter = 0
    
    for wifi in list_wifi:
        for wf in wifi:
            index_wifi = Wifi_sorted.index(str(wf[0]))
      #      print index_wifi
            wifi_matrix[counter,index_wifi+2] = int(wf[1])
        counter += 1

    
    wifi_matrix1  = np.zeros([int(test['row_id'].count()),len(allWifi_sorted)+2])
    wifi_matrix1[:,0:2] = test_loc
    counter1 = 0
    for wifi1 in list_wifi1:
        for wf1 in wifi1:
     #   print wf[1]
            index_wifi1 = Wifi_sorted.index(str(wf1[0]))
    #    print index_wifi
            wifi_matrix1[counter1,index_wifi1+2] = int(wf1[1])
        counter1 += 1
    label = list(data['shop_id'])
   
    wifi_matrix[:,0:2] = wifi_matrix[:,0:2]/100.0
    wifi_matrix[:,0:2] = wifi_matrix[:,0:2]*0.68
    wifi_matrix1[:,0:2] = wifi_matrix1[:,0:2]/100.0
    wifi_matrix1[:,0:2] = wifi_matrix1[:,0:2] *0.75

    #---------------------Predict----------------------------
    knn = KNeighborsClassifier(n_jobs= -1)
    knn.fit(wifi_matrix,label)
    predict = knn.predict(wifi_matrix1)
    predict = pd.DataFrame(predict,columns=['shop_id'])

    predict = pd.DataFrame(predict,columns=['shop_id'])
    pre_result = pd.concat([test['row_id'],predict['shop_id']],axis = 1)
    result = pd.DataFrame(columns=['row_id','shop_id'])
    result = pd.concat([result,pre_result],axis = 0)
    print mall
    result.to_csv('resuat_'+mall+'.csv',index=None)
#result.to_csv('result.csv',index = 0)





