import numpy as np
from sklearn.model_selection import train_test_split

def train_val_test_split(n_inputs_per_patient, data_cancer, data_healthy):
    # shuffle input instances by patient
    n_cancer_patients = data_cancer.shape[0] // n_inputs_per_patient
    n_healthy_patients = data_healthy.shape[0] // n_inputs_per_patient

    N = n_inputs_per_patient
    # shuffle cancer 
    M,n,p = n_cancer_patients, data_cancer.shape[-2], data_cancer.shape[-1]
    np.random.shuffle(data_cancer.reshape(M,-1,n,p))
    # shuffle healthy
    M,n,p = n_healthy_patients, data_healthy.shape[-2], data_healthy.shape[-1]
    np.random.shuffle(data_healthy.reshape(M,-1,n,p))

    # leave out 20% of each cancer/healthy patients for test set
    cancer_split_idx = int(0.8 * n_cancer_patients) * n_inputs_per_patient
    healthy_split_idx = int(0.8 * n_healthy_patients) * n_inputs_per_patient

    # split training data sets
    data_train = np.concatenate((data_cancer[:cancer_split_idx], 
                                data_healthy[:healthy_split_idx]), axis=0)
    data_test = np.concatenate((data_cancer[cancer_split_idx:], 
                                data_healthy[healthy_split_idx:]), axis=0)

    # generate labels
    labels_train = np.concatenate((np.ones(cancer_split_idx), 
                                np.zeros(healthy_split_idx)))
    labels_test = np.concatenate((np.ones(data_cancer.shape[0] - cancer_split_idx), 
                                np.zeros(data_healthy.shape[0] - healthy_split_idx)))

    # shuffle/split train data into train & validation
    data_train, data_val, \
    labels_train, labels_val = train_test_split(data_train, labels_train, 
                                                test_size=0.20,
                                                random_state=42)

    return (data_train, data_val, data_test, labels_train, labels_val, labels_test)