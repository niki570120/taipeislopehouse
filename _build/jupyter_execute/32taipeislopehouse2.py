#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
df=pd.read_csv('./32taipeislopehouse.csv', encoding='cp950')
df


# In[33]:


df = df.drop(['Unnamed: 14','Unnamed: 15','Unnamed: 16','Unnamed: 17','Unnamed: 18','Unnamed: 19'], axis=1)
df


# In[34]:


df = df.drop(['disaster'], axis=1)
df


# In[35]:


df1=df.drop([32])
df1


# In[36]:


df1.info()


# In[37]:


df2 = df1.drop(['coordinates'], axis=1)
df2


# In[38]:


from sklearn.model_selection import train_test_split
features = ['historicaldisaster','monitoring','abnormal','downhillslope','sensitivearea']

X = df2[features]
y = df2['DOWN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[9]:


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.image as pltimg
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier

import numpy as np 
import pandas as pd 

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
data = tree.export_graphviz(dtree, out_file=None)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('32taipeislopehouse2-1.png')

img=pltimg.imread('32taipeislopehouse2-1.png')
imgplot = plt.imshow(img)
plt.show()


# In[10]:


X_train


# In[11]:


X_test


# In[12]:


y_train


# In[13]:


y_test


# In[14]:


from sklearn.linear_model import LogisticRegression#邏輯式回歸
clf = LogisticRegression()

clf.fit(X_train, y_train)


# In[15]:


clf.score(X_train, y_train)


# In[16]:


clf.predict(X)


# In[17]:


from sklearn.linear_model import LinearRegression#線性回歸
clf1 = LinearRegression()

clf1.fit(X_train, y_train)


# In[18]:


clf1.predict(X)


# In[20]:


from sklearn.linear_model import Perceptron#類神經網路
clf2 = Perceptron()

clf2.fit(X_train, y_train)


# In[21]:


clf2.predict(X)


# In[22]:


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.image as pltimg
from sklearn import tree
import pydotplus

from sklearn.ensemble import RandomForestClassifier # 隨機森林
clf3 = RandomForestClassifier()

clf3= tree.export_graphviz(dtree, out_file=None)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('32taipeislpoehouse2-2.png')

img=pltimg.imread('32taipeislpoehouse2-2.png')
imgplot = plt.imshow(img)
plt.show()


# In[23]:


y_train


# In[25]:


y_test


# In[26]:


from sklearn.neighbors import KNeighborsClassifier # k近鄰算法
clf4 = KNeighborsClassifier()

clf4.fit(X_train, y_train)


# In[27]:


clf4.predict(X)


# In[28]:


from sklearn.svm import SVC #支持向量機
clf5 = SVC()

clf5.fit(X_train, y_train)


# In[29]:


clf5.predict(X)


# In[30]:


from sklearn.neural_network import MLPClassifier#類神經網路mlp分類器
clf6 = MLPClassifier()

clf6.fit(X_train, y_train)


# In[31]:


clf6.predict(X)


# In[ ]:





# In[ ]:




