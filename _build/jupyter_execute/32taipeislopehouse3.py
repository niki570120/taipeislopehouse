#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
df=pd.read_csv('./32taipeislopehouse.csv', encoding='cp950')
df


# In[28]:


df = df.drop(['Unnamed: 14','Unnamed: 15','Unnamed: 16','Unnamed: 17','Unnamed: 18','Unnamed: 19'], axis=1)
df


# In[29]:


df = df.drop(['disaster'], axis=1)
df


# In[30]:


df1=df.drop([32])
df1


# In[31]:


df2 = df1.drop(['LOU','coordinates'], axis=1)
df2


# In[32]:


df2.AREA = df2.AREA.map({'信義':0,'北投':1, '士林':2,'大安':3,'中山':4,'內湖':5,'南港':6,'文山':7})
df2


# In[33]:


from sklearn.model_selection import train_test_split

y = df2.DOWN
X = df2.drop('DOWN', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[34]:


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
graph.write_png('32taipeislopehouse3-1.png')

img=pltimg.imread('32taipeislopehouse3-1.png')
imgplot = plt.imshow(img)
plt.show()


# In[35]:


y_train


# In[36]:


y_test


# In[37]:


from sklearn.linear_model import LogisticRegression#邏輯式回歸
clf = LogisticRegression()

clf.fit(X_train, y_train)


# In[38]:


clf.score(X_train, y_train)


# In[39]:


clf.predict(X)


# In[40]:


from sklearn.linear_model import LinearRegression#線性回歸
clf1 = LinearRegression()

clf1.fit(X_train, y_train)


# In[41]:


clf1.predict(X)


# In[42]:


from sklearn.linear_model import Perceptron#類神經網路
clf2 = Perceptron()

clf2.fit(X_train, y_train)


# In[43]:


clf2.predict(X)


# In[44]:


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
graph.write_png('32taipeislpoehouse2.png')

img=pltimg.imread('32taipeislpoehouse2.png')
imgplot = plt.imshow(img)
plt.show()


# In[45]:


y_train


# In[46]:


y_test


# In[47]:


from sklearn.neighbors import KNeighborsClassifier # k近鄰算法
clf4 = KNeighborsClassifier()

clf4.fit(X_train, y_train)


# In[48]:


clf4.predict(X)


# In[49]:


from sklearn.svm import SVC #支持向量機
clf5 = SVC()

clf5.fit(X_train, y_train)


# In[50]:


clf5.predict(X)


# In[51]:


from sklearn.neural_network import MLPClassifier#類神經網路mlp分類器
clf6 = MLPClassifier()

clf6.fit(X_train, y_train)


# In[52]:


clf6.predict(X)


# In[ ]:





# In[ ]:




