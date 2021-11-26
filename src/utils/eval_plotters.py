import numpy as np
import matplotlib.pyplot as plt

# plotting function to track loss/accuracy over epochs
def plot_loss_acc(iters, hist, save_path=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    # plot training/validation loss over epochs:
    ax1.plot(np.arange(iters) + 1, hist['loss'])
    ax1.plot(np.arange(iters) + 1, hist['val_loss'])
    ax1.legend(["Training", "Validation"])
    ax1.set_title("Training/validation loss per epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    # plot training/validation accuracy over epochs:
    ax2.plot(np.arange(iters) + 1, hist['accuracy'])
    ax2.plot(np.arange(iters) + 1, hist['val_accuracy'])
    ax2.legend(["Training", "Validation"])
    ax2.set_title("Training/validation accuracy per epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    if save_path is not None:
        fig.savefig(save_path)


def plot_ROC(fpr, tpr, labels_val, preds, save_path=None):
    fig = plt.figure(figsize=(6,6))  # plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.title("AUC: %.4f" %metrics.roc_auc_score(labels_val, preds))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if save_path is not None:
        fig.savefig(save_path)