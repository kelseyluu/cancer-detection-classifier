import numpy as np
import h5py
import pandas as pd

# load and reshape cancer data 
data_cancer = pd.read_csv("./cancer/processed/cancer_samples.txt", 
                          sep='\t', header=None)
data_cancer = np.array(data_cancer)
data_cancer = np.reshape(data_cancer, (-1, 1, 1000, 2))

# load and reshape healthy data
data_healthy = pd.read_csv("./healthy/processed/healthy_samples.txt", 
                          sep='\t', header=None)
data_healthy = np.array(data_healthy)
data_healthy = np.reshape(data_healthy, (-1, 1, 1000, 2))

# save to h5 file
h5f = h5py.File('./data.h5', 'w')
h5f.create_dataset('cancer', data=data_cancer)
h5f.create_dataset('healthy', data=data_healthy)
h5f.close()