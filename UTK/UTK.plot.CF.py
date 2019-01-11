import itertools
import numpy as np
import matplotlib.pyplot as plt

import importlib
from matplotlib2tikz import save as tikz_save

# UTK Classes
class_names = [
    "Walk",
    "Sit Down",
    "Stand Up",
    "Pick Up",
    "Carry",
    "Throw",
    "Push",
    "Pull",
    "Wave Hands",
    "Clap Hands"
]
# UTK CNN All Augmentations AVERAGE CONFUSION MATRIX Runs = 5
title = "UTK CNN All Augmentations"
png_file = "utk.cnn.allaug.png"
cnf_matrix = np.array(
    [[71.00, 02.00, 00.00, 00.00, 19.00, 03.00, 00.00, 00.00, 00.00, 05.00],
     [00.00, 94.00, 00.00, 06.00, 00.00, 00.00, 00.00, 00.00, 00.00, 00.00],
     [00.00, 05.00, 75.00, 20.00, 00.00, 00.00, 00.00, 00.00, 00.00, 00.00],
     [01.00, 16.00, 01.00, 82.00, 00.00, 00.00, 00.00, 00.00, 00.00, 00.00],
     [24.00, 04.00, 00.00, 01.00, 50.00, 01.00, 01.00, 01.00, 00.00, 18.00],
     [00.00, 00.00, 00.00, 00.00, 03.00, 42.00, 17.00, 25.00, 03.00, 10.00],
     [00.00, 00.00, 00.00, 00.00, 04.00, 03.00, 67.00, 20.00, 03.00, 03.00],
     [00.00, 00.00, 00.00, 00.00, 00.00, 17.00, 18.00, 35.00, 10.00, 20.00],
     [00.00, 00.00, 00.00, 00.00, 00.00, 00.00, 00.00, 03.00, 91.00, 06.00],
     [00.00, 00.00, 00.00, 00.00, 12.00, 07.00, 00.00, 08.00, 15.00, 58.00]]
)


# Plot normalized confusion matrix
plt.figure()
cmap = plt.cm.Blues
plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap, aspect='auto')
plt.title(title)
# plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
cm = cnf_matrix
fmt = '.2f'  # if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
             horizontalalignment="center", ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')

# plt.show()
plt.savefig(png_file, bbox_inches='tight', dpi=600)
# tikz_save("test.tex", figureheight='4cm', figurewidth='6cm')
# plt.clf()


