import numpy as np
import scipy.io as sio

DIR = "C:\\Users\\tuwie\\PycharmProjects\\acin\\UTK\\"

key = "kfold0"
DATASET_NAME = "UTK"

train_data_file = DIR + DATASET_NAME + ".train." + key + ".data.npy"
train_labels_file = DIR + DATASET_NAME + ".train." + key + ".labels.npy"

train_data = np.load(train_data_file)
train_label = np.load(train_labels_file)

# Leaving random label picking for later, if needed
UTK_actions_map = {
    "walk": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "sitdown": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "standup": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "pickup": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "carry": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "throw": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "push": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "pull": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "wavehands": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "claphands": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

exportedActionsUTK = {}
export_filename = "utk_exporta.mat"
for key in UTK_actions_map.keys():
    for seq_id in range(train_data.shape[0]):
        if np.array_equal(train_label[seq_id, :], UTK_actions_map.get(key)):
            print("Matched " + key)
            sequence = train_data[seq_id, :, :, :]
            clean_length = int(sequence[~np.all(sequence == 0.0, axis=2)].shape[0]/sequence.shape[1])
            sequence_clean = sequence[~np.all(sequence == 0.0, axis=2)] \
                .reshape([clean_length, sequence.shape[1], sequence.shape[2]])
            exportedActionsUTK[key] = sequence_clean
            break
print("Saving exported actions in MATLAB file " + "utk_exporta.mat")
sio.savemat(export_filename, exportedActionsUTK)

