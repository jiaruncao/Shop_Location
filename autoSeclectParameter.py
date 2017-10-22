
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os


# In[30]:


knn = KNeighborsClassifier()
for r,_,files in os.walk('splitData/'):                               #自行修改为当前目录名
    for file in files:
        print file
        data = pd.read_csv(os.path.join('splitData/',file))           #自行修改为当前目录名
        location = data['longitude']
        location = pd.DataFrame(location)
        location.insert(column= 'latitude',value=data['latitude'],loc = 0)
        location= np.array(location)
        label = pd.DataFrame(data['shop_id'])
        label = np.array(label)
        knn = KNeighborsClassifier()
        label = label.reshape(label.shape[0],)
        k_range = list(range(1,20))
        leaf_range = list(range(1,2))
        weight_options = ['uniform','distance']
        algorithm_options = ['auto','ball_tree','kd_tree','brute']
        param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
        gridKNN = GridSearchCV(knn,param_gridknn,cv=3,scoring='accuracy',verbose=1,error_score= 0)
        gridKNN.fit(location,label)
        fr = open('parameter.txt','a')
        fr.write(str(file)+'\n')
        fr.write(str(gridKNN.best_score_)+'\n')
        fr.write(str(gridKNN.best_params_)+'\n')
        fr.write('\n')
        fr.close()
        






