
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os


# In[26]:


#加载训练集和测试集
eva = pd.read_csv('AB榜测试集-evaluation_public.csv')
data = pd.read_csv('user_shop_behavior_cleanned')
shop = pd.read_csv('shop_info_cleanned')
fr = open('myParameter.txt')


# In[27]:


#训练模型
i = 1
knnList = []
for r,_,files in os.walk('mySplitData/'):
    for file in sorted(files):
        data = pd.read_csv(os.path.join('mySplitData/',file))
     #   line = fr.next().strip('\n')
     #  line = fr.next()
        
        line = fr.readline()
        if len(line) == 0: # Zero length indicates EOF
            break
        line = line.strip('{')
        line = line.strip('}')
        line1 = line.split('\'')
        weights_value = line1[5]
        line1[2] = line1[2].strip(':')
        line1[2] = line1[2].strip(' ')
        num = line1[2].split(',')
        num = int(num[0])
        location = data['longitude']
        location = pd.DataFrame(location)
        location.insert(column= 'latitude',value=data['latitude'],loc = 0)
        location= np.array(location)
        label = pd.DataFrame(data['shop_id'])
        label = np.array(label)
        label = label.reshape(label.shape[0],)
        knn = KNeighborsClassifier(n_neighbors=num,weights= weights_value)
        knn.fit(location,label)
        knnList.append(knn)
        print (i)
        i += 1
fr.close()


# In[10]:


'''
malls=pd.DataFrame(shop['mall_id'])
malls=malls[['mall_id']]
malls.drop_duplicates(inplace=True)
malls.count()
'''


'''
test = eva
malls=malls.mall_id.values.tolist()
print('malls type:',type(malls[0]))
#为train训练集添加mall_id
shop=shop[['shop_id','mall_id']]
shop.drop_duplicates(inplace=True)
#test=pd.merge(test,shop,on=['shop_id'],how='left')
#test.drop_duplicates(inplace=True)
#防止数据集过大，切分数据集
print('split dataset finished')
for mall in malls:
    test_t=test[test['mall_id']==mall]
    test_t.to_csv(mall+'.csv',index=None)
'''


# In[32]:

#预测testData，并将预测值按mall_id存入predictList
predictList = []
i = 0

for r,_,files in os.walk('splitTestData/'):
    for file in sorted(files):
        print file
        testData = pd.read_csv(os.path.join('splitTestData/',file))
        testLocation = testData['longitude']
        testLocation = pd.DataFrame(testLocation)
        testLocation.insert(column= 'latitude',value=testData['latitude'],loc = 0)
        testLocation= np.array(testLocation)
        predict = knnList[i].predict(testLocation)
        predictList.append(predict)
        print i
        i += 1


# In[58]:

#统计predictList中的预测数
count = 0
for t in range(97):
    count = len(predictList[t]) + count
count

# In[59]:
#拼接数据

i = 0
result = pd.DataFrame(columns=['row_id','shop_id'])
for r,_,files in os.walk('splitTestData/'):
    for file in sorted(files):
        combineData_row = pd.read_csv(os.path.join('splitTestData/',file))
        combineData_row = combineData_row['row_id']
        combineData_shop = pd.DataFrame(predictList[i],columns=['shop_id'])
        okData = pd.concat([combineData_row,combineData_shop],axis = 1)
        result =pd.concat([result,okData],ignore_index=True)
        i += 1
        
        


# In[63]:

#导出result
result.to_csv('result.csv',index = 0)





