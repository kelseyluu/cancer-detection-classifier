import tensorflow as tf
import itertools as it
import numpy as np
import pandas as pd
from train_model import cnn_train

# Hyper-parameter search function
def grid_search(training_dataset,
                validation_dataset,
                dropout_rates,
                reg_coeffs,
                learning_rates,
                pbars,
                save_path=None,
                verbose=True,
                **kwargs):
    """performs a grid hyperparameter search with loss as the metric
    
    perform a grid search over different hyper parameter configurations
    to determing the optimal configuration for the cnn
    
    Arguments:
      dropout_rates: a list of dropout_rates to test
      reg_coeffs: a list or L2 lambda terms to test
      learning_rates: a list of learning rates to test
      pbars: a list of `tqdm` progress bar objects to use for displaying
        grid search and training progress
        * len(pbars) == 3
      verbose: (False)flag to print grid search parameters and epoch loss on 
        each respective iteration
      **kwargs: any additional valid keyword arguments for passing to
        `cnn_train` 
        
    Returns:
      losses_df: a `pd.DataFrame` table of the grid search results
        * len(losses_df) == (len(dropout_rates) * len(reg_coeffs) 
                             * len(learning_rates))
    """
    losses = []
    hyperparam_combos = list(
        it.product(dropout_rates, reg_coeffs, learning_rates))
        
    pbars[0].reset(len(hyperparam_combos))
    pbars[0].set_description('Grid Search')
    for dropout_rate, reg_coeff, learning_rate in hyperparam_combos:
        if verbose:
            tf.print('Beginning training for dr={:.3f}, l2_lambda={:.1e}, '
                     'lr={:.3f}'.format(dropout_rate, reg_coeff, learning_rate))
        
        hist = cnn_train(training_dataset, 
                 validation_dataset,
                 reg_coeff = reg_coeff, 
                 dropout_rate = dropout_rate,
                 learning_rate = learning_rate,
                 batch_size = 128, 
                 num_epochs = 10,
                 verbose = True)
                 
        best_loss = np.min(hist.get('val_loss'))

        print("Best val loss: %.4f" %best_loss)
        losses.append([dropout_rate, 
                       reg_coeff, 
                       learning_rate, 
                       best_loss])
        
        pbars[0].update()
        
    losses_df = pd.DataFrame(losses, 
                             columns=['dropout rate',
                                      'L2 lambda',
                                      'learning rate',
                                      'validation loss'])                                              

    if save_path is not None:
        grid_search_pivot = (losses_df.loc[losses_df['learning rate'] != 0.1]
                      .pivot_table(values=['validation loss'],
                                    columns=['L2 lambda'],
                                    index=['learning rate', 'dropout rate']))
        grid_search_pivot.style.format('{:.3f}').background_gradient(cmap='magma_r',
                                                              axis=None).to_excel(save_path)

    return losses_df
