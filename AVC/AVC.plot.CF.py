import itertools
import numpy as np
import matplotlib.pyplot as plt

import importlib
from matplotlib2tikz import save as tikz_save

# ACIN Classes
class_names = [
    "Grab",
    "Move",
    "Place",
    "Reach"
]

# ACIN CNN NO Augmentations Runs = 3
# title = "ACIN CNN No Augmentations"
# png_file = "acin.cnn.noaug.png"
# cnf_matrix = np.array(
#     [[67.38, 08.14, 09.16, 15.32],
#      [05.21, 79.03, 05.90, 09.87],
#      [09.70, 06.15, 50.73, 33.41],
#      [14.09, 08.49, 04.02, 73.40]]
# )

title = "ACIN CNN All Augmentations"
png_file = "acin.cnn.allaug.png"
cnf_matrix = np.array(
    [[67.79, 11.95, 08.54, 11.72],
     [03.38, 84.75, 06.51, 05.36],
     [08.73, 07.93, 52.10, 31.23],
     [11.25, 11.82, 06.40, 70.53]]
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


