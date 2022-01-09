#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
df=pd.read_csv('./32taipeislopehouse.csv', encoding='cp950')
df


# In[62]:


df.info()


# In[63]:


df = df.drop(['Unnamed: 14','Unnamed: 15','Unnamed: 16','Unnamed: 17','Unnamed: 18','Unnamed: 19'], axis=1)
df


# In[64]:


df = df.drop(['disaster'], axis=1)
df


# In[65]:


df1=df.drop([32])
df1


# In[66]:


df1.info()


# In[67]:


df2 = df1.drop(['coordinates'], axis=1)
df2


# In[68]:


from sklearn.model_selection import train_test_split
features = ['Buildingnumber', 'households', 'persons', 'historicaldisaster','monitoring','abnormal','downhillslope','sensitivearea']

X = df2[features]
y = df2['DOWN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[69]:


print(X_train)
print(y_train)


# In[70]:


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
graph.write_png('32taipeislopehouse1-1.png')

img=pltimg.imread('32taipeislopehouse1-1.png')
imgplot = plt.imshow(img)
plt.show()


# In[71]:


y_train


# In[72]:


y_test


# In[73]:


from sklearn.linear_model import LogisticRegression#邏輯式回歸
clf = LogisticRegression()

clf.fit(X_train, y_train)


# In[74]:


clf.intercept_


# In[75]:


clf.score(X_train, y_train)


# In[76]:


clf.predict(X)


# In[77]:


from sklearn.linear_model import LinearRegression#線性回歸
clf1 = LinearRegression()

clf1.fit(X_train, y_train)


# In[78]:


clf1.predict(X)


# In[79]:


clf1.intercept_


# In[80]:


from sklearn.linear_model import Perceptron#類神經網路
clf2 = Perceptron()

clf2.fit(X_train, y_train)


# In[81]:


clf2.predict(X)


# In[82]:


clf2.intercept_


# In[83]:


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
graph.write_png('32taipeislpoehouse1-2.png')

img=pltimg.imread('32taipeislpoehouse1-2.png')
imgplot = plt.imshow(img)
plt.show()


# In[84]:


y_train


# In[85]:


y_test


# In[86]:


from sklearn.neighbors import KNeighborsClassifier # k近鄰算法
clf4 = KNeighborsClassifier()

clf4.fit(X_train, y_train)


# In[87]:


clf4.predict(X)


# In[88]:


from sklearn.svm import SVC #支持向量機
clf5 = SVC()

clf5.fit(X_train, y_train)


# In[89]:


clf5.predict(X)


# In[90]:


from sklearn.neural_network import MLPClassifier#類神經網路mlp分類器
clf6 = MLPClassifier()

clf6.fit(X_train, y_train)


# In[91]:


clf6.predict(X)


# In[92]:


# 預測#邏輯式回歸
y_pred = clf.predict(X_test)
y_pred


# In[93]:


# 幫模型打分數#邏輯式回歸
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[94]:


# 幫模型打分數
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[95]:


# 幫模型打分數#邏輯式回歸
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[96]:


# 預測#線性回歸
y_pred = clf1.predict(X_test)
y_pred


# In[97]:


# 幫模型打分數#線性回歸
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[98]:


# 幫模型打分數#線性回歸
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[99]:


# 預測# k近鄰算法
y_pred = clf4.predict(X_test)
y_pred


# In[100]:


# 幫模型打分數# k近鄰算法
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[101]:


# 幫模型打分數# k近鄰算法
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[102]:


# 幫模型打分數
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[103]:


import numpy as np

np.sum((y_test - y_pred) ** 2) / y_test.shape[0]


# In[104]:


# 預測#類神經網路
y_pred = clf2.predict(X_test)
y_pred


# In[105]:


# 幫模型打分數#類神經網路
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[106]:


# 幫模型打分數#類神經網路
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[107]:


# 幫模型打分數
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[108]:


# 預測#支持向量機
y_pred = clf5.predict(X_test)
y_pred


# In[109]:


# 幫模型打分數#支持向量機
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[110]:


# 幫模型打分數#支持向量機
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[111]:


# 幫模型打分數
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[112]:


# 預測#類神經網路mlp分類器
y_pred = clf6.predict(X_test)
y_pred


# In[113]:


# 幫模型打分數#類神經網路mlp分類器
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[114]:


# 幫模型打分數#類神經網路mlp分類器
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[115]:


# 幫模型打分數
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[116]:


# 幫模型打分數
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[117]:


# 預測#決策樹
y_pred = df2.predict(X_test)
y_pred


# In[ ]:





# In[ ]:




