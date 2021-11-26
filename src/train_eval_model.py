#### Installations ####
# general python utilities
import os
import random
import functools

# data-science & processing tools
import numpy as np
import pandas as pd
import h5py
from sklearn import metrics
from sklearn.metrics import accuracy_score

# tensorflow
import tensorflow as tf
import tensorflow.keras as K

# required TensorFlow version >= 2.0.0
tf_version = tf.__version__
assert int(tf_version[0]) >= 2, "Tensorflow version must be >= 2.0"

# plotting utilities
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 18})

# progress bar
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm_notebook as tqdm
    
# import custom utils
from utils.split_data import train_val_test_split
from utils.training_utils import cnn_train
from utils.grid_search import grid_search
from utils.eval_plotters import plot_loss_acc, plot_ROC

# seed random numbers for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
DEFAULT_DTYPE = 'float32'



#### Load and process data (272 cancer, 262 healthy) #### 
f = h5py.File('../data/data.h5', 'r')
data_cancer = np.array(f['cancer'])
data_healthy = np.array(f['healthy'])

# shuffle and sample input instances by patient for test set, 
# by feature for validation set
n_inputs_per_patient = 487
(data_train, data_val, data_test, \
  labels_train, labels_val, labels_test) = train_val_test_split(n_inputs_per_patient, 
                                                                data_cancer, 
                                                                data_healthy)


    

#### Hyperparameter search ####
pbar = functools.partial(tqdm, leave=True, ncols='70%')
pbars = [pbar() for _ in range(3)]

# define parameter grid
dropout_rates = [0.8, 0.5, 0.2]
reg_coeffs = [1e-03, 1e-06, 0]
learning_rates = [0.1, 0.01, 1e-3, 1e-4]

# perform search and save results
grid_search_df = grid_search((data_train, labels_train), 
                              (data_val, labels_val),
                              dropout_rates, 
                              reg_coeffs, 
                              learning_rates, 
                              pbars,
                              save_path="./grid_search.csv")

min_row_index = grid_search_df['validation loss'].idxmin()
min_row_df = grid_search_df.loc[min_row_index]
optimal_hyperparams = dict(dropout_rate=min_row_df['dropout rate'],
                           reg_coeff=min_row_df['L2 lambda'],
                           learning_rate=min_row_df['learning rate'])



#### Implement training with optimal hyperparameters ####
SAVE_MODEL = False
optimal_hyperparams = {'dropout_rate': 0.2, 
                       'reg_coeff': 1e-06, 
                       'learning_rate': 0.0001}
model, hist = cnn_train(training_dataset = (data_train, labels_train), 
                        validation_dataset = (data_val, labels_val),
                        **optimal_hyperparams,
                        batch_size = 128, 
                        num_epochs = 50,
                        do_save_model = SAVE_MODEL,
                        verbose = True)

# save plot of model loss over epochs
plot_loss_acc(len(hist.get('val_loss')), hist, 
              save_path = "./train_val_loss_history.png")



#### Evaluate model on test set ####
preds = model.predict(data_test)  
num_patients = int(np.floor(preds.shape[0]/n_inputs_per_patient))
weight_grid = np.linspace(0,1,1000,endpoint=False)
cnn_test_preds = []
for patient in range(num_patients):  # generate patient level predictions
  range_begin = (patient * n_inputs_per_patient) % \
                (preds.shape[0] - n_inputs_per_patient)
  range_end = range_begin + n_inputs_per_patient
  patient_preds = np.array(preds[range_begin:range_end,:])
  # calculate probability of cancer based on patient_preds (feature predictions)
  log_likelihood = [
    np.log(weight_guess*patient_preds + \
          (1-weight_guess)*(1-patient_preds)).sum()
    for weight_guess in weight_grid
  ]
  est_ratio = weight_grid[np.argmax(log_likelihood)]
  cnn_test_preds.append(1 if est_ratio > 0 else 0)

# report performance metrics, save AUC plot
patient_test_labels = labels_test[::n_inputs_per_patient]
cnn_df = {'true_label': patient_test_labels, 'cnn_pred': cnn_test_preds}
print(f'Patient level test set accuracy: \
      {accuracy_score(cnn_df.true_label, cnn_df.cnn_pred)}')
fpr, tpr, _ = metrics.roc_curve(patient_test_labels, cnn_test_preds) 
plot_ROC(fpr, tpr, patient_test_labels, cnn_test_preds, 
        save_path = "./ROC_plot.png")
