
# coding: utf-8

# In[70]:

import numpy as np


# In[ ]:

def conv_to_csv(dataset, name):
    path = '../../data/'+dataset
    fpath = path+'/'+name
    print fpath
    
    with open(fpath,'r') as f:
        data = f.read().split('\n')
        if len(data[-1]) == 0:
            data.pop()
        
    l=data[0]
    l=l.split(' ')
    
    mat2 = np.zeros((int(l[0]),int(l[1])))
    
    for line in data[1:]:
        l = line.split(' ')
        l = [float(x) for x in l]
        ind = int(l[0])
        mat2[ind,:] = l[1:]    
        
    print('Number of data points :: %s ' % mat2.shape[0])
    print('Number of data points :: %s ' % mat2.shape[1])
    np.savetxt(fpath+'.csv',mat2,fmt='%.6e')

