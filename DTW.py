#!/usr/bin/env python
# coding: utf-8

# In[19]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install fastdtw')


# In[33]:


import numpy as np
import pandas as pd
import time
start = time.time()


# In[21]:


df = pd.read_table(r"C:\Users\gupye\OneDrive\바탕 화면\온라인교육_논문\activity_accumulator_std_test(0320) (1).csv", sep = '|' ,encoding = 'UTF8')
df = df.drop([938212], axis = 0) # 이상한 데이터 삭제


# In[22]:


#학번이 숫자로만 이루어진 학생들만 filter
df['digit'] = df.user_id.str.isnumeric()
df = df[df['digit'].isin([True])]
df


# In[23]:


#count
test = df.groupby(['user_id','timestamp']).count().reset_index()
test = test[['user_id','timestamp','data']].sort_values(by  = 'user_id', ascending = True)
test['timestamp'] = pd.to_datetime(test['timestamp'])
test = test.pivot(index = 'timestamp', columns = 'user_id', values = 'data')

print("time :", time.time() - start)


# In[24]:


# 시간 단위로 Downsampling
mi = test.resample(rule = '1H').sum().transpose()

mi
print("time :", time.time() - start)


# In[35]:


# matrix normalization
from sklearn.preprocessing import StandardScaler

matrix = np.array(mi, dtype = np.int32)

# 10개만 일단 테스트
matrix1 = matrix[:10]
matrix1 = StandardScaler().fit_transform(matrix1) #z-normalization

print("time :", time.time() - start)
matrix1


# In[36]:


#DTW 함수정의
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
def dtw_dist(x, y):
    distance, path = fastdtw(x, y, dist = euclidean)
    return distance

print("time :", time.time() - start)


# In[27]:


dist_pairs = []

for i in range(len(matrix1)):
    for j in range(len(matrix1)):
        dist = dtw_dist(matrix1[i], matrix1[j])
        dist_pairs.append((i,j,dist))
        
print("time :", time.time() - start)


# In[28]:


dist_frame = pd.DataFrame(dist_pairs, columns = ['A','B','Dist'])
sf = dist_frame[dist_frame['Dist']>0].sort_values(['A','B','Dist']).reset_index(drop=1)
sfe = sf[sf['A']<sf['B']]
sfe


# In[29]:


# 시각화 확인
import matplotlib as mpl
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


plt.figure(figsize = (50, 20))
plt.plot(matrix1[0])


# In[31]:


plt.figure(figsize = (50, 20))
plt.plot(matrix1[1])


# In[ ]:


#결과 저장
sfe.to_csv(r"C:\Users\gupye\OneDrive\바탕 화면\온라인교육_논문\result.csv", encoding = 'utf8')

