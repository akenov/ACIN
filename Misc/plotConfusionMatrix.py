import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Code from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, model_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("plots\\" + model_name + '_confusion_matrix.png', bbox_inches='tight', dpi=600)
    plt.clf()
    return


# Manually set the confusion matrixes to be plotted ATM

# RNN2 Data from 29.06.2018
RNN2_ConfMat = np.array([
    [11135, 875,  162],
    [401, 12885,   94],
    [100,    46, 5902]
])
# RNN3 Data from 29.06.2018
RNN3_ConfMat = np.array([
    [11865,  189, 118],
    [1003, 12250, 127],
    [246,      5, 5797]
])
# CNN1 from 23.07.2018
CNN1_ConfMat = np.array([
    [9175,  0, 0],
    [19, 14559, 0],
    [6,   0, 7841]
])
class_names = np.array(["GRAB", "MOVEOBJ", "REACH"], dtype=object)

# cnf_matrix = RNN2_ConfMat
# modelName = "RNN2 (15 epochs)"

# cnf_matrix = RNN3_ConfMat
# modelName = "RNN3 (15 epochs)"

cnf_matrix = CNN1_ConfMat
modelName = "CNN1 (100 epochs)"

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, model_name=modelName,
                      normalize=True, title='Normalized confusion matrix ' + modelName)
# plt.show()

