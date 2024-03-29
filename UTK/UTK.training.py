from __future__ import print_function

import pickle
import math
import os
import numpy as np
import itertools
from numpy import matlib
from datetime import datetime
from DataGenerator import DataGenerator
import keras
from keras import regularizers
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Permute, Activation
from keras.layers import Conv2D, MaxPooling2D, LSTM, Masking, BatchNormalization
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

# NOTE: Dataset from http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html is of bad quality.
# FrameIDs are missing, doubled or false described in actionLabel
# Values are corrected to authors best knowledge


def extend_sequences(sequence_, avg_length=10):
    sequence_length = sequence_.shape[0]
    ext_factors = [0, 0.25, 0.5, 0.75]
    ext_step = len(ext_factors)
    new_sequence_ = np.zeros([(sequence_length-1) * ext_step + 1, sequence_.shape[1]])
    if sequence_length < avg_length:
        for frame_id in range(int(sequence_length-1)):
            frm_current = sequence_[frame_id, :]
            frm_next = sequence_[int(frame_id + 1), :]
            frm_step = np.subtract(frm_next, frm_current)
            for ext_id in range(ext_step):
                new_sequence_[int(frame_id * ext_step + ext_id), :] = \
                    np.add(frm_current, np.multiply(frm_step, ext_factors[ext_id]))
        return new_sequence_
    else:
        return sequence_


def skeleton_reshape(sequence_):
    new_sequence = np.zeros([MAX_WIDTH, NUM_JOINTS, 3])
    sequence_length = sequence_.shape[0]
    for frame_id in range(sequence_length):
        for joint in range(NUM_JOINTS):
            new_sequence[frame_id, joint, :] = sequence_[frame_id, 3*joint: 3*joint + 3]
    return new_sequence


def sliding_window_generator(sample):
    window_size = math.floor(COEFF_SLIDINGWINDOW * sample.shape[0])
    slided_window_samples = []
    for frame in range(sample.shape[0] - window_size + 1):
        window = sample[frame: frame+window_size, :]
        slided_window_samples.append(window)
    return slided_window_samples


def partial_fit_data_to_scaler(sample_name):
    sample_ = np.loadtxt(sample_name)
    sample_data = sample_[:, 1:]
    sample_ids = np.array(sample_[:, 0]).astype(int).reshape([-1, 1])

    walk_params = content[line + 1].split(" ")
    walk_min = int(np.where(sample_ids == int(walk_params[1].lstrip()))[0])
    walk_max = int(np.where(sample_ids == int(walk_params[2].lstrip()))[0]+1)
    scalerStd.partial_fit(sample_data[walk_min: walk_max, :])
    scalerMinMax.partial_fit(sample_data[walk_min: walk_max, :])

    sitdown_params = content[line + 2].split(" ")
    sitdown_min = int(np.where(sample_ids == int(sitdown_params[1].lstrip()))[0])
    sitdown_max = int(np.where(sample_ids == int(sitdown_params[2].lstrip()))[0]+1)
    scalerStd.partial_fit(sample_data[sitdown_min: sitdown_max, :])
    scalerMinMax.partial_fit(sample_data[sitdown_min: sitdown_max, :])

    standup_params = content[line + 3].split(" ")
    standup_min = int(np.where(sample_ids == int(standup_params[1].lstrip()))[0])
    standup_max = int(np.where(sample_ids == int(standup_params[2].lstrip()))[0]+1)
    scalerStd.partial_fit(sample_data[standup_min: standup_max, :])
    scalerMinMax.partial_fit(sample_data[standup_min: standup_max, :])

    pickup_params = content[line + 4].split(" ")
    pickup_min = int(np.where(sample_ids == int(pickup_params[1].lstrip()))[0])
    pickup_max = int(np.where(sample_ids == int(pickup_params[2].lstrip()))[0]+1)
    scalerStd.partial_fit(sample_data[pickup_min: pickup_max, :])
    scalerMinMax.partial_fit(sample_data[pickup_min: pickup_max, :])

    carry_params = content[line + 5].split(" ")
    if 'NaN' in carry_params:
        print("Carry Value NaN detected. Filling in blanks.")
    else:
        carry_min = int(np.where(sample_ids == int(carry_params[1].lstrip()))[0])
        carry_max = int(np.where(sample_ids == int(carry_params[2].lstrip()))[0] + 1)
        scalerStd.partial_fit(sample_data[carry_min: carry_max, :])
        scalerMinMax.partial_fit(sample_data[carry_min: carry_max, :])

    throw_params = content[line + 6].split(" ")
    throw_min = int(np.where(sample_ids == int(throw_params[1].lstrip()))[0])
    throw_max = int(np.where(sample_ids == int(throw_params[2].lstrip()))[0]+1)
    scalerStd.partial_fit(sample_data[throw_min: throw_max, :])
    scalerMinMax.partial_fit(sample_data[throw_min: throw_max, :])

    push_params = content[line + 7].split(" ")
    push_min = int(np.where(sample_ids == int(push_params[1].lstrip()))[0])
    push_max = int(np.where(sample_ids == int(push_params[2].lstrip()))[0]+1)
    scalerStd.partial_fit(sample_data[push_min: push_max, :])
    scalerMinMax.partial_fit(sample_data[push_min: push_max, :])

    pull_params = content[line + 8].split(" ")
    pull_min = int(np.where(sample_ids == int(pull_params[1].lstrip()))[0])
    pull_max = int(np.where(sample_ids == int(pull_params[2].lstrip()))[0]+1)
    scalerStd.partial_fit(sample_data[pull_min: pull_max, :])
    scalerMinMax.partial_fit(sample_data[pull_min: pull_max, :])

    wavehands_params = content[line + 9].split(" ")
    wavehands_min = int(np.where(sample_ids == int(wavehands_params[1].lstrip()))[0])
    wavehands_max = int(np.where(sample_ids == int(wavehands_params[2].lstrip()))[0]+1)
    scalerStd.partial_fit(sample_data[wavehands_min: wavehands_max, :])
    scalerMinMax.partial_fit(sample_data[wavehands_min: wavehands_max, :])

    claphands_params = content[line + 10].split(" ")
    claphands_min = int(np.where(sample_ids == int(claphands_params[1].lstrip()))[0])
    claphands_max = int(np.where(sample_ids == int(claphands_params[2].lstrip()))[0]+1)
    scalerStd.partial_fit(sample_data[claphands_min: claphands_max, :])
    scalerMinMax.partial_fit(sample_data[claphands_min: claphands_max, :])


def process_sample(sample_name):
    sample_ = np.loadtxt(sample_name)
    sample_data = sample_[:, 1:]
    sample_ids = np.array(sample_[:, 0]).astype(int).reshape([-1, 1])
    actions_map = {
        "walk": 0,
        "sitdown": 1,
        "standup": 2,
        "pickup": 3,
        "carry": 4,
        "throw": 5,
        "push": 6,
        "pull": 7,
        "wavehands": 8,
        "claphands": 9
    }
    data_set = []
    labels = []

    walk_params = content[line + 1].split(" ")
    walk_min = int(np.where(sample_ids == int(walk_params[1].lstrip()))[0])
    walk_max = int(np.where(sample_ids == int(walk_params[2].lstrip()))[0]+1)
    walk_sample = sample_data[walk_min: walk_max, :]
    if EXTEND_ACTIONS:
        walk_sample = extend_sequences(walk_sample, FRAMES_THRESHOLD)
    if USE_SCALER:
        walk_mean = np.mean(walk_sample, axis=0).reshape(1, -1)
        walk_mean_arr = matlib.repmat(walk_mean, walk_sample.shape[0], 1)
        walk_sample = np.subtract(walk_sample, walk_mean_arr)
        # walk_var = np.var(walk_sample, axis=0).reshape(1, -1)
        # walk_var = walk_var + np.multiply(np.ones(walk_var.shape), k.epsilon())
        # walk_sigma = np.sqrt(walk_var)
        # walk_sample = np.divide(walk_sample, walk_sigma)
        # print(scalerMinMax)
        # walk_sample = scalerStd.transform(walk_sample)
        # walk_sample = scalerMinMax.transform(walk_sample)

    data_set.append(skeleton_reshape(walk_sample))
    labels.append(actions_map.get("walk"))
    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(walk_sample):
            data_set.append(skeleton_reshape(windowed_sample))
            labels.append(actions_map.get("walk"))

    sitdown_params = content[line + 2].split(" ")
    sitdown_min = int(np.where(sample_ids == int(sitdown_params[1].lstrip()))[0])
    sitdown_max = int(np.where(sample_ids == int(sitdown_params[2].lstrip()))[0]+1)
    sitdown_sample = sample_data[sitdown_min: sitdown_max, :]
    if EXTEND_ACTIONS:
        sitdown_sample = extend_sequences(sitdown_sample, FRAMES_THRESHOLD)
    if USE_SCALER:
        sitdown_mean = np.mean(sitdown_sample, axis=0).reshape(1, -1)
        sitdown_mean_arr = matlib.repmat(sitdown_mean, sitdown_sample.shape[0], 1)
        sitdown_sample = np.subtract(sitdown_sample, sitdown_mean_arr)
        # sitdown_sample = scalerStd.transform(sitdown_sample)
        # sitdown_sample = scalerMinMax.transform(sitdown_sample)

    data_set.append(skeleton_reshape(sitdown_sample))
    labels.append(actions_map.get("sitdown"))
    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(sitdown_sample):
            data_set.append(skeleton_reshape(windowed_sample))
            labels.append(actions_map.get("sitdown"))

    standup_params = content[line + 3].split(" ")
    standup_min = int(np.where(sample_ids == int(standup_params[1].lstrip()))[0])
    standup_max = int(np.where(sample_ids == int(standup_params[2].lstrip()))[0]+1)
    standup_sample = sample_data[standup_min: standup_max, :]
    if EXTEND_ACTIONS:
        standup_sample = extend_sequences(standup_sample, FRAMES_THRESHOLD)
    if USE_SCALER:
        standup_mean = np.mean(standup_sample, axis=0).reshape(1, -1)
        standup_mean_arr = matlib.repmat(standup_mean, standup_sample.shape[0], 1)
        standup_sample = np.subtract(standup_sample, standup_mean_arr)
        # standup_sample = scalerStd.transform(standup_sample)
        # standup_sample = scalerMinMax.transform(standup_sample)

    data_set.append(skeleton_reshape(standup_sample))
    labels.append(actions_map.get("standup"))
    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(standup_sample):
            data_set.append(skeleton_reshape(windowed_sample))
            labels.append(actions_map.get("standup"))

    pickup_params = content[line + 4].split(" ")
    pickup_min = int(np.where(sample_ids == int(pickup_params[1].lstrip()))[0])
    pickup_max = int(np.where(sample_ids == int(pickup_params[2].lstrip()))[0]+1)
    pickup_sample = sample_data[pickup_min: pickup_max, :]
    if EXTEND_ACTIONS:
        pickup_sample = extend_sequences(pickup_sample, FRAMES_THRESHOLD)
    if USE_SCALER:
        pickup_mean = np.mean(pickup_sample, axis=0).reshape(1, -1)
        pickup_mean_arr = matlib.repmat(pickup_mean, pickup_sample.shape[0], 1)
        pickup_sample = np.subtract(pickup_sample, pickup_mean_arr)
        # pickup_sample = scalerStd.transform(pickup_sample)
        # pickup_sample = scalerMinMax.transform(pickup_sample)

    data_set.append(skeleton_reshape(pickup_sample))
    labels.append(actions_map.get("pickup"))
    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(pickup_sample):
            data_set.append(skeleton_reshape(windowed_sample))
            labels.append(actions_map.get("pickup"))

    carry_params = content[line + 5].split(" ")
    if 'NaN' in carry_params:
        print("Carry Value NaN detected. Filling in blanks.")
        data_set.append(np.zeros([MAX_WIDTH, 20, 3]))
        labels.append(actions_map.get("carry"))
    else:
        carry_min = int(np.where(sample_ids == int(carry_params[1].lstrip()))[0])
        carry_max = int(np.where(sample_ids == int(carry_params[2].lstrip()))[0] + 1)
        carry_sample = sample_data[carry_min: carry_max, :]
        if EXTEND_ACTIONS:
            carry_sample = extend_sequences(carry_sample, FRAMES_THRESHOLD)
        if USE_SCALER:
            carry_mean = np.mean(carry_sample, axis=0).reshape(1, -1)
            carry_mean_arr = matlib.repmat(carry_mean, carry_sample.shape[0], 1)
            carry_sample = np.subtract(carry_sample, carry_mean_arr)
            # carry_sample = scalerStd.transform(carry_sample)
            # carry_sample = scalerMinMax.transform(carry_sample)

        data_set.append(skeleton_reshape(carry_sample))
        labels.append(actions_map.get("carry"))
        if USE_SLIDINGWINDOW:
            for windowed_sample in sliding_window_generator(carry_sample):
                data_set.append(skeleton_reshape(windowed_sample))
                labels.append(actions_map.get("carry"))

    throw_params = content[line + 6].split(" ")
    throw_min = int(np.where(sample_ids == int(throw_params[1].lstrip()))[0])
    throw_max = int(np.where(sample_ids == int(throw_params[2].lstrip()))[0]+1)
    throw_sample = sample_data[throw_min: throw_max, :]
    if EXTEND_ACTIONS:
        throw_sample = extend_sequences(throw_sample, FRAMES_THRESHOLD)
    if USE_SCALER:
        throw_mean = np.mean(throw_sample, axis=0).reshape(1, -1)
        throw_mean_arr = matlib.repmat(throw_mean, throw_sample.shape[0], 1)
        throw_sample = np.subtract(throw_sample, throw_mean_arr)
        # throw_sample = scalerStd.transform(throw_sample)
        # throw_sample = scalerMinMax.transform(throw_sample)

    data_set.append(skeleton_reshape(throw_sample))
    labels.append(actions_map.get("throw"))
    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(throw_sample):
            data_set.append(skeleton_reshape(windowed_sample))
            labels.append(actions_map.get("throw"))

    push_params = content[line + 7].split(" ")
    push_min = int(np.where(sample_ids == int(push_params[1].lstrip()))[0])
    push_max = int(np.where(sample_ids == int(push_params[2].lstrip()))[0]+1)
    push_sample = sample_data[push_min: push_max, :]
    if EXTEND_ACTIONS:
        push_sample = extend_sequences(push_sample, FRAMES_THRESHOLD)
    if USE_SCALER:
        push_mean = np.mean(push_sample, axis=0).reshape(1, -1)
        push_mean_arr = matlib.repmat(push_mean, push_sample.shape[0], 1)
        push_sample = np.subtract(push_sample, push_mean_arr)
        # push_sample = scalerStd.transform(push_sample)
        # push_sample = scalerMinMax.transform(push_sample)

    data_set.append(skeleton_reshape(push_sample))
    labels.append(actions_map.get("push"))
    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(push_sample):
            data_set.append(skeleton_reshape(windowed_sample))
            labels.append(actions_map.get("push"))

    pull_params = content[line + 8].split(" ")
    pull_min = int(np.where(sample_ids == int(pull_params[1].lstrip()))[0])
    pull_max = int(np.where(sample_ids == int(pull_params[2].lstrip()))[0]+1)
    pull_sample = sample_data[pull_min: pull_max, :]
    if EXTEND_ACTIONS:
        pull_sample = extend_sequences(pull_sample, FRAMES_THRESHOLD)
    if USE_SCALER:
        pull_mean = np.mean(pull_sample, axis=0).reshape(1, -1)
        pull_mean_arr = matlib.repmat(pull_mean, pull_sample.shape[0], 1)
        pull_sample = np.subtract(pull_sample, pull_mean_arr)
        # pull_sample = scalerStd.transform(pull_sample)
        # pull_sample = scalerMinMax.transform(pull_sample)

    data_set.append(skeleton_reshape(pull_sample))
    labels.append(actions_map.get("pull"))
    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(pull_sample):
            data_set.append(skeleton_reshape(windowed_sample))
            labels.append(actions_map.get("pull"))

    wavehands_params = content[line + 9].split(" ")
    wavehands_min = int(np.where(sample_ids == int(wavehands_params[1].lstrip()))[0])
    wavehands_max = int(np.where(sample_ids == int(wavehands_params[2].lstrip()))[0]+1)
    wavehands_sample = sample_data[wavehands_min: wavehands_max, :]
    if EXTEND_ACTIONS:
        wavehands_sample = extend_sequences(wavehands_sample, FRAMES_THRESHOLD)
    if USE_SCALER:
        wavehands_mean = np.mean(wavehands_sample, axis=0).reshape(1, -1)
        wavehands_mean_arr = matlib.repmat(wavehands_mean, wavehands_sample.shape[0], 1)
        wavehands_sample = np.subtract(wavehands_sample, wavehands_mean_arr)
        # wavehands_sample = scalerStd.transform(wavehands_sample)
        # wavehands_sample = scalerMinMax.transform(wavehands_sample)

    data_set.append(skeleton_reshape(wavehands_sample))
    labels.append(actions_map.get("wavehands"))
    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(wavehands_sample):
            data_set.append(skeleton_reshape(windowed_sample))
            labels.append(actions_map.get("wavehands"))

    claphands_params = content[line + 10].split(" ")
    claphands_min = int(np.where(sample_ids == int(claphands_params[1].lstrip()))[0])
    claphands_max = int(np.where(sample_ids == int(claphands_params[2].lstrip()))[0]+1)
    claphands_sample = sample_data[claphands_min: claphands_max, :]
    if EXTEND_ACTIONS:
        claphands_sample = extend_sequences(claphands_sample, FRAMES_THRESHOLD)
    if USE_SCALER:
        claphands_mean = np.mean(claphands_sample, axis=0).reshape(1, -1)
        claphands_mean_arr = matlib.repmat(claphands_mean, claphands_sample.shape[0], 1)
        claphands_sample = np.subtract(claphands_sample, claphands_mean_arr)
        # claphands_sample = scalerStd.transform(claphands_sample)
        # claphands_sample = scalerMinMax.transform(claphands_sample)

    data_set.append(skeleton_reshape(claphands_sample))
    labels.append(actions_map.get("claphands"))
    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(claphands_sample):
            data_set.append(skeleton_reshape(windowed_sample))
            labels.append(actions_map.get("claphands"))

    return np.asarray(data_set), np.asarray(labels).reshape([-1, 1])


def run_keras_cnn_model(loso_, run_suffix):
    modelname = DATASET_NAME + ' CNN LOSO #' + loso_[4:]

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type = 'train.' + loso_
    histsave = RESULTS_DIR + DATASET_NAME + '_mnistcnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    scoresave = RESULTS_DIR + DATASET_NAME + '_mnistcnn_scores.' + loso_ + '.' + run_suffix + '.save'
    weightsave = RESULTS_DIR + DATASET_NAME + '_mnistcnn_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = RESULTS_DIR + DATASET_NAME + '_mnistcnn_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    print("Test Data Shape = %s " % (test_data_.shape,))
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(len(AUGMENTATIONS) * train_data_.shape[0])
    batch_size_aug = len(AUGMENTATIONS) * CNN_BATCH_SIZE
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    print("Input Shape = %s " % (ishape, ))

    tensorboard = TensorBoard(log_dir=RESULTS_DIR, histogram_freq=0,
                              write_graph=True, write_images=True)
    training_generator = DataGenerator(DATASET_NAME, generator_type, batch_size_aug,
                                       ishape, list_idxes, AUGMENTATIONS)

    cnn_model = Sequential()

    cnn_model.add(BatchNormalization(input_shape=ishape))
    cnn_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same', use_bias=False))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(COEFF_DROPOUT))
    cnn_model.add(Dense(300, activation='relu', use_bias=False))
    cnn_model.add(Dropout(COEFF_DROPOUT))
    cnn_model.add(Dense(100, activation='relu', use_bias=False))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(NUM_CLASSES, activation='softmax'))

    cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=get_optimizer(),
                      metrics=['accuracy'])

    cnn_model.summary()
    print(datetime.now())
    print("Start training")
    history = cnn_model.fit_generator(generator=training_generator,
                                      epochs=NUM_EPOCHS, validation_data=(test_data_, test_labels_),
                                      shuffle=False, use_multiprocessing=MULTI_CPU,
                                      callbacks=[tensorboard])

    # history = model.fit(train_data, train_labels,
    #                     batch_size=batch_size_aug,
    #                     epochs=epochs_n, verbose=1,
    #                     validation_data=(test_data, test_labels))

    print(datetime.now())
    print(test_data_.shape)
    scores = cnn_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug)
    print(datetime.now())

    print("# KERAS MODEL: " + modelname + " # # # ")
    print('Test loss: %.4f' % scores[0])
    print('Test accuracy: %.3f %%' % (scores[1] * 100))
    RESULTS.append(scores[1] * 100)
    pred_labels = cnn_model.predict(test_data_, batch_size=batch_size_aug)
    # print("Prediction matrix data:")
    # print(pred_labels.shape)
    # print(pred_labels)
    # print(datetime.now())

    with open(scoresave, 'wb') as file_pi:
        pickle.dump(scores, file_pi)
        print("Saved training history %s" % scoresave)

    with open(histsave, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        print("Saved training history %s" % histsave)

    cnn_model.save(weightsave)
    print("Saved model weights to %s" % weightsave)

    cnf_matrix = confusion_matrix(test_labels_.argmax(axis=1), pred_labels.argmax(axis=1))
    cnf_matrix_proc = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix_proc = np.multiply(cnf_matrix_proc, 100)
    print(" CONFUSION MATRIX in % ")
    print(cnf_matrix_proc)
    with open(cfsave, 'wb') as file_pi:
        pickle.dump(cnf_matrix_proc, file_pi)
        print("Saved confusion matrix to %s" % cfsave)
    print("# KERAS MODEL: " + modelname + " # # # ")
    print("Closing Keras/TF Session")
    keras.backend.clear_session()
    return


def run_keras_lstm_model(loso_, run_suffix):
    modelname = DATASET_NAME + ' LSTM LOSO #' + loso_[4:]

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = RESULTS_DIR + DATASET_NAME + '_lstm_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    scoresave = RESULTS_DIR + DATASET_NAME + '_lstm_scores.' + loso_ + '.' + run_suffix + '.save'
    weightsave = RESULTS_DIR + DATASET_NAME + '_lstm_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = RESULTS_DIR + DATASET_NAME + '_lstm_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(AUGMENTATIONS) * train_data.shape[0], 1)
    batch_size_aug = len(AUGMENTATIONS) * RNN_BATCH_SIZE
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    print("Input Shape = %s " % (ishape,))

    training_generator = DataGenerator(DATASET_NAME, generator_type_train,
                                       batch_size_aug, ishape, list_idxes, AUGMENTATIONS)

    tensorboard = TensorBoard(log_dir=RESULTS_DIR, histogram_freq=0,
                              write_graph=True, write_images=True)

    lstm_model = Sequential()
    resh_shape = (test_data_.shape[1], test_data_.shape[2] * test_data_.shape[3])
    lstm_model.add(Reshape(resh_shape, input_shape=ishape))
    lstm_model.add(Masking(mask_value=0.0, input_shape=lstm_model.layers[-1].output_shape))
    lstm_model.add(BatchNormalization(axis=2))
    lstm_model.add(LSTM(100, kernel_regularizer=regularizers.l2(COEFF_REGULARIZATION_L2),
                        stateful=False, use_bias=False, dropout=COEFF_DROPOUT))
    lstm_model.add(Dense(NUM_CLASSES, activation='softmax'))

    lstm_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=get_optimizer(),
                       metrics=['accuracy'])

    lstm_model.summary()
    print(datetime.now())
    print("Start training")

    # history = lstm_model.fit(train_data, train_labels,
    #                          batch_size=batch_size_aug,
    #                          epochs=1, verbose=1,
    #                          validation_data=(test_data, test_labels))

    history = lstm_model.fit_generator(generator=training_generator,
                                       epochs=NUM_EPOCHS, validation_data=(test_data_, test_labels_),
                                       shuffle=False, use_multiprocessing=MULTI_CPU,
                                       callbacks=[tensorboard])

    print(datetime.now())
    print(test_data_.shape)
    scores = lstm_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug)
    print(datetime.now())

    print("# KERAS MODEL: " + modelname + " # # # ")
    print('Test loss: %.4f' % scores[0])
    print('Test accuracy: %.3f %%' % (scores[1] * 100))
    RESULTS.append(scores[1] * 100)
    pred_labels = lstm_model.predict(test_data_, batch_size=batch_size_aug)
    # print("Prediction matrix data:")
    # print(pred_labels.shape)
    # print(pred_labels)
    # print(datetime.now())

    with open(scoresave, 'wb') as file_pi:
        pickle.dump(scores, file_pi)
        print("Saved training history %s" % scoresave)

    with open(histsave, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        print("Saved training history %s" % histsave)

    lstm_model.save(weightsave)
    print("Saved model weights to %s" % weightsave)

    cnf_matrix = confusion_matrix(test_labels_.argmax(axis=1), pred_labels.argmax(axis=1))
    cnf_matrix_proc = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix_proc = np.multiply(cnf_matrix_proc, 100)
    print(" CONFUSION MATRIX in % ")
    print(cnf_matrix_proc)
    with open(cfsave, 'wb') as file_pi:
        pickle.dump(cnf_matrix_proc, file_pi)
        print("Saved confusion matrix to %s" % cfsave)
    print("# KERAS MODEL: " + modelname + " # # # ")
    print("Closing Keras/TF Session")
    keras.backend.clear_session()
    return


def run_keras_nunez_model(loso_, run_suffix):
    modelname = DATASET_NAME + ' Nunez LOSO #' + loso_[4:]

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    cnn_histsave = RESULTS_DIR + DATASET_NAME + '_nunez_cnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    rnn_histsave = RESULTS_DIR + DATASET_NAME + '_nunez_rnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    cnn_scoresave = RESULTS_DIR + DATASET_NAME + '_nunez_cnn_scores.' + loso_ + '.' + run_suffix + '.save'
    rnn_scoresave = RESULTS_DIR + DATASET_NAME + '_nunez_rnn_scores.' + loso_ + '.' + run_suffix + '.save'
    rnn_weightsave = RESULTS_DIR + DATASET_NAME + '_nunez_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = RESULTS_DIR + DATASET_NAME + '_nunez_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    print("Test Data Shape = %s " % (test_data_.shape,))
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(len(AUGMENTATIONS) * train_data_.shape[0])
    batch_size_aug_cnn = len(AUGMENTATIONS) * CNN_BATCH_SIZE
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    print("Input Shape = %s " % (ishape, ))

    tensorboard = TensorBoard(log_dir=RESULTS_DIR, histogram_freq=0,
                              write_graph=True, write_images=True)
    training_generator_cnn = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug_cnn,
                                           ishape, list_idxes, AUGMENTATIONS)

    conv_model = Sequential()
    conv_model.add(BatchNormalization(input_shape=ishape))
    conv_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same', use_bias=False))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))

    # CNN part
    conv_model.add(Dropout(COEFF_DROPOUT))
    conv_model.add(Dense(300, activation='relu', use_bias=False))
    conv_model.add(Dropout(COEFF_DROPOUT))
    conv_model.add(Dense(100, activation='relu', use_bias=False))
    conv_model.add(Flatten())
    conv_model.add(Dense(NUM_CLASSES, activation='softmax'))

    conv_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=get_optimizer(),
                       metrics=['accuracy'])

    conv_model.summary()
    print(datetime.now())
    print("Start training")
    history_cnn = conv_model.fit_generator(generator=training_generator_cnn,
                                           epochs=NUM_EPOCHS, validation_data=(test_data_, test_labels_),
                                           shuffle=False, use_multiprocessing=MULTI_CPU,
                                           callbacks=[tensorboard])

    print(datetime.now())
    cnn_scores = conv_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug_cnn)
    print(datetime.now())
    print('CNN Test loss: %.4f' % cnn_scores[0])
    print('CNN Test accuracy: %.3f %%' % (cnn_scores[1] * 100))
    CNN_RESULTS.append(cnn_scores[1] * 100)

    with open(cnn_scoresave, 'wb') as file_pi:
        pickle.dump(cnn_scores, file_pi)
        print("Saved CNN scores in %s" % cnn_scoresave)

    with open(cnn_histsave, 'wb') as file_pi:
        pickle.dump(history_cnn.history, file_pi)
        print("Saved CNN training history in %s" % cnn_histsave)

    print("Continue with RNN")
    conv_model.layers.pop()  # Dense(Softmax)
    conv_model.layers.pop()  # Flatten()
    conv_model.layers.pop()  # Dense(100)
    conv_model.layers.pop()  # Dense(300)

    # RNN part
    batch_size_aug_rnn = len(AUGMENTATIONS) * RNN_BATCH_SIZE
    training_generator_rnn = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug_rnn,
                                           ishape, list_idxes, AUGMENTATIONS)

    nunez_model = Sequential()
    nunez_model.add(BatchNormalization(input_shape=ishape))
    nunez_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False,
                           trainable=CNN_TRAINABLE, kernel_regularizer=regularizers.l2(COEFF_REGULARIZATION_L2)))
    nunez_model.add(MaxPooling2D(pool_size=(2, 2), trainable=CNN_TRAINABLE))
    nunez_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same', use_bias=False,
                           trainable=CNN_TRAINABLE, kernel_regularizer=regularizers.l2(COEFF_REGULARIZATION_L2)))
    nunez_model.add(MaxPooling2D(pool_size=(2, 2), trainable=CNN_TRAINABLE))
    nunez_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False,
                           trainable=CNN_TRAINABLE, kernel_regularizer=regularizers.l2(COEFF_REGULARIZATION_L2)))
    nunez_model.add(MaxPooling2D(pool_size=(2, 2), trainable=CNN_TRAINABLE))

    nunez_model.set_weights(conv_model.get_weights())

    # nunez_model.add(Permute((2, 1, 3))) # no need in the Nunez shape style
    # print(nunez_model.layers[-1].output_shape)
    nunez_model.add(Reshape((15, 200)))
    # print(nunez_model.layers[-1].output_shape)
    nunez_model.add(Masking(mask_value=0.0, input_shape=nunez_model.layers[-1].output_shape))
    nunez_model.add(BatchNormalization())
    nunez_model.add(LSTM(100, kernel_regularizer=regularizers.l2(COEFF_REGULARIZATION_L2), stateful=False,
                         use_bias=False, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_RECCURENT_DROPOUT))
    nunez_model.add(Dropout(COEFF_DROPOUT))
    # nunez_model.add(Flatten())
    nunez_model.add(Dense(NUM_CLASSES, activation='softmax'))

    nunez_model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=get_optimizer(),
                        metrics=['accuracy'])

    nunez_model.summary()
    print(datetime.now())
    print("Start training")
    history_rnn = nunez_model.fit_generator(generator=training_generator_rnn,
                                            epochs=int(NUM_EPOCHS*5), validation_data=(test_data_, test_labels_),
                                            shuffle=False, use_multiprocessing=MULTI_CPU,
                                            callbacks=[tensorboard])

    print(datetime.now())
    rnn_scores = nunez_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug_cnn)
    print(datetime.now())

    print("# KERAS MODEL: " + modelname + " # # # ")
    print('CNN+RNN Test loss: %.4f' % rnn_scores[0])
    print('CNN+RNN Test accuracy: %.3f %%' % (rnn_scores[1] * 100))
    RESULTS.append(rnn_scores[1] * 100)
    pred_labels = nunez_model.predict(test_data_, batch_size=batch_size_aug_cnn)
    # print("Prediction matrix data:")
    # print(pred_labels.shape)
    # print(pred_labels)
    print(datetime.now())

    with open(rnn_scoresave, 'wb') as file_pi:
        pickle.dump(rnn_scores, file_pi)
        print("Saved RNN scores in %s" % rnn_scoresave)

    with open(rnn_histsave, 'wb') as file_pi:
        pickle.dump(history_rnn.history, file_pi)
        print("Saved RNN training history in %s" % rnn_histsave)

    nunez_model.save(rnn_weightsave)
    print("Saved model weights to %s" % rnn_weightsave)

    cnf_matrix = confusion_matrix(test_labels_.argmax(axis=1), pred_labels.argmax(axis=1))
    cnf_matrix_proc = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix_proc = np.multiply(cnf_matrix_proc, 100)
    print(" CONFUSION MATRIX in % ")
    print(cnf_matrix_proc)
    with open(cfsave, 'wb') as file_pi:
        pickle.dump(cnf_matrix_proc, file_pi)
        print("Saved confusion matrix to %s" % cfsave)
    print("# KERAS MODEL: " + modelname + " # # # ")
    print("Closing Keras/TF Session")
    keras.backend.clear_session()
    return


def trim_to_batch(nonbatch_data, nonbatch_labels, batchsize):
    print("Original array length: " + str(nonbatch_data.shape[0]))
    newlength_data = math.floor(nonbatch_data.shape[0] / batchsize) * batchsize
    print("New batch-sized array length: " + str(newlength_data))
    return nonbatch_data[:newlength_data, :], nonbatch_labels[:newlength_data, :]


def get_optimizer():
    print("Using " + OPTIMIZER[0] + " optimizer")
    if OPTIMIZER[0] == "Adam":
        return keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
    elif OPTIMIZER[0] == "AdaDelta":
        return keras.optimizers.Adadelta(lr=0.1, rho=0.993, decay=0.0)


def print_summary():
    results_len = len(RESULTS)
    results_arr = np.asarray(RESULTS)
    cnn_results_arr = np.asarray(CNN_RESULTS)
    print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")
    print("| AUGMENTATIONS: %s" % AUGMENTATIONS)
    print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")
    print("| EXTEND_ACTIONS: " + str(EXTEND_ACTIONS))
    print("| USE_SLIDINGWINDOW: " + str(USE_SLIDINGWINDOW))
    print("| USE_SCALER: " + str(USE_SCALER))
    print("| CNN_TRAINABLE: " + str(CNN_TRAINABLE))
    print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")
    print("| OPTIMIZER: " + OPTIMIZER[0])
    print("| NUM_EPOCHS: " + str(NUM_EPOCHS))
    print("| CNN_BATCH_SIZE: " + str(CNN_BATCH_SIZE))
    print("| RNN_BATCH_SIZE: " + str(RNN_BATCH_SIZE))
    print("| FRAMES_THRESHOLD: " + str(FRAMES_THRESHOLD))
    print("| COEFF_SLIDINGWINDOW: " + str(COEFF_SLIDINGWINDOW))
    print("| COEFF_DROPOUT: " + str(COEFF_DROPOUT))
    print("| COEFF_REGULARIZATION_L2: " + str(COEFF_REGULARIZATION_L2))
    print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")
    print("| " + DATASET_NAME + " " + model + " AVERAGE ACCURACY %.2f " % (np.sum(results_arr)/results_len))
    print("| Final Single Results ")
    print("| ", end=" ")
    for res in RESULTS:
        print("%.2f |" % res, end=" ")
    print("")
    if TRAIN_MODELS[0] == "ConvRNN":
        print("| CNN AVERAGE ACCURACY %.2f " % (np.sum(cnn_results_arr)/results_len))
        print("| CNN Single Results ")
        print("| ", end=" ")
        for res in CNN_RESULTS:
            print("%.2f |" % res, end=" ")
        print("")
    print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")


sample_names = [
    's01_e01',
    's01_e02',
    's02_e01',
    's02_e02',
    's03_e01',
    's03_e02',
    's04_e01',
    's04_e02',
    's05_e01',
    's05_e02',
    's06_e01',
    's06_e02',
    's07_e01',
    's07_e02',
    's08_e01',
    's08_e02',
    's09_e01',
    's09_e02',
    's10_e01',
    's10_e02'
]

batch_names = [
    ('kfold0', 's01_e01'),
    ('kfold0', 's01_e02'),
    ('kfold0', 's02_e01'),
    ('kfold0', 's02_e02'),
    ('kfold2', 's03_e01'),
    ('kfold2', 's03_e02'),
    ('kfold2', 's04_e01'),
    ('kfold2', 's04_e02'),
    ('kfold4', 's05_e01'),
    ('kfold4', 's05_e02'),
    ('kfold4', 's06_e01'),
    ('kfold4', 's06_e02'),
    ('kfold6', 's07_e01'),
    ('kfold6', 's07_e02'),
    ('kfold6', 's08_e01'),
    ('kfold6', 's08_e02'),
    ('kfold8', 's09_e01'),
    ('kfold8', 's09_e02'),
    ('kfold8', 's10_e01'),
    ('kfold8', 's10_e02')
]

DATASET_NAME = 'UTK'
# STATIC PARAMETERS
NUM_FILES = 20
LINE_STEP = 11
NUM_CLASSES = 10
MAX_WIDTH = 120
NUM_JOINTS = 20
# EDITABLE PARAMETERS
# DIRECTORY = "./joints/"
# UTKLABELSFILE = "./actionLabel.txt"
# MULTI_CPU = True
# DIRECTORY = "/home/antonk/racer/UTKinect3D/joints/"
# UTKLABELSFILE = "/home/antonk/racer/UTKinect3D/actionLabel.txt"
DIRECTORY = "D:\\!DA-20092018\\UTKinectAction3D\\joints\\"
UTKLABELSFILE = "D:\\!DA-20092018\\UTKinectAction3D\\actionLabel.txt"
MULTI_CPU = False
# SET OUTPUT_SAVES OUTSIDE THE DOCKER CONTAINER
OUTPUT_SAVES = "./"
EXTEND_ACTIONS = True
USE_SLIDINGWINDOW = True
USE_SCALER = False
CNN_TRAINABLE = True
FRAMES_THRESHOLD = 13
COEFF_SLIDINGWINDOW = 0.8
COEFF_DROPOUT = 0.6
LSTM_DROPOUT = 0.2
LSTM_RECCURENT_DROPOUT = 0.2
COEFF_REGULARIZATION_L2 = 0.015
CNN_BATCH_SIZE = 50
RNN_BATCH_SIZE = 16
k.set_epsilon(1e-06)

ITERATIONS = 1
NUM_EPOCHS = 100
AUGMENTATIONS = [
    # 'none',
    # "scale_shift",
    # 'scale',
    # 'shift_gauss_xy',
    'noise',
    # 'subsample',
    # 'interpol',
    # 'translate',
    # 'scale_translate'
]
OPTIMIZER = [
    "Adam",
    # "AdaDelta"
]
TRAIN_MODELS = [
    'CNN',
    # 'LSTM',
    # 'ConvRNN'
]
# END OF PARAMETERS

le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
scalerStd = StandardScaler()
scalerMinMax = MinMaxScaler(feature_range=(-1, 1))

actionLabels = open(UTKLABELSFILE, "r")
content = actionLabels.readlines()
actionLabels.close()

RESULTS_DIR = OUTPUT_SAVES + DATASET_NAME + "." + datetime.today().strftime('%d-%m-%Y_%H%M') + "/"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

for model in TRAIN_MODELS:
    for run_num in np.arange(ITERATIONS):
        RESULTS = []
        CNN_RESULTS = []
        for key, batch_group in itertools.groupby(batch_names, lambda x: x[0]):
            #     #     #      #      #      #      #      #      #      #      #
            # SPEED UP RELATIVE EVAL: @JB,@MATT - USE SINGLE SPLIT
            if key != "kfold0":
                print("Skipping split " + key)
                continue
            #     #     #      #      #      #      #      #      #      #      #

            print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")
            print("| File Batch: " + key)
            print("| Augmentations: %s" % AUGMENTATIONS)
            print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")

            if EXTEND_ACTIONS:
                print("Extend Actions #Frames Threshold: %d" % FRAMES_THRESHOLD)
            if USE_SLIDINGWINDOW:
                print("Sliding Window Length: %.2f" % COEFF_SLIDINGWINDOW)

            train_data_file = DATASET_NAME + ".train." + key + ".data"
            train_labels_file = DATASET_NAME + ".train." + key + ".labels"
            test_data_file = DATASET_NAME + ".test." + key + ".data"
            test_labels_file = DATASET_NAME + ".test." + key + ".labels"

            if os.path.exists(train_data_file + ".npy") and os.path.exists(train_labels_file + ".npy") \
                    and os.path.exists(test_data_file + ".npy") and os.path.exists(test_labels_file + ".npy"):
                print("Sample TRAIN and TEST data sets files found. Skipping generation from skeleton data.")
            else:
                data_train = np.zeros([1, MAX_WIDTH, 20, 3])
                labels_train = np.zeros([1, 1])
                data_test = np.zeros([1, MAX_WIDTH, 20, 3])
                labels_test = np.zeros([1, 1])

                test_files = []
                for batch_pair in batch_group:
                    test_files.append(batch_pair[1])
                train_files = sample_names[:]
                for testfile in test_files:
                    train_files.remove(testfile)
                print(test_files)
                print("- - - - - - - - - - - -")
                print(train_files)

                if USE_SCALER:
                    print("Sipping through the data fitting the scaler")
                    for line in range(0, 220, 11):
                        # print("Parsing start line: %d " % l)
                        filename = DIRECTORY + "joints_" + content[line].strip('\n').lstrip() + ".txt"
                        partial_fit_data_to_scaler(filename)

                for line in range(0, 220, 11):
                    # print("Parsing start line: %d " % l)
                    filename = DIRECTORY + "joints_" + content[line].strip('\n').lstrip() + ".txt"
                    print("Parsing file: %s " % filename)
                    file_data, file_labels = process_sample(filename)

                    if any(test in filename for test in train_files):
                        data_train = np.append(data_train, file_data, axis=0)
                        labels_train = np.append(labels_train, file_labels, axis=0)
                    else:
                        data_test = np.append(data_test, file_data, axis=0)
                        labels_test = np.append(labels_test, file_labels, axis=0)

                # print(le.fit(np.asarray(labels_train)).classes_)
                utk_dataset_train = data_train[1:, :, :, :]
                feat_labenc = le.fit_transform(labels_train[1:, :])
                utk_labels_train = ohe.fit_transform(feat_labenc.reshape(len(feat_labenc), 1))

                utk_dataset_test = data_test[1:, :, :, :]
                feat_labenc = le.fit_transform(labels_test[1:, :])
                utk_labels_test = ohe.fit_transform(feat_labenc.reshape(len(feat_labenc), 1))

                utk_dataset_train, utk_labels_train = shuffle(utk_dataset_train, utk_labels_train, random_state=42)
                utk_dataset_test, utk_labels_test = shuffle(utk_dataset_test, utk_labels_test, random_state=42)

                print("Saving data & labels to files: ")
                print("TRAIN length: %d samples" % utk_dataset_train.shape[0])
                print("TRAIN data  : %s" % train_data_file)
                print("TRAIN labels: %s" % train_labels_file)
                np.save(train_data_file, utk_dataset_train)
                np.save(train_labels_file, utk_labels_train)
                print("TEST length: %d samples" % utk_dataset_test.shape[0])
                print("TEST data  : %s" % test_data_file)
                print("TEST labels: %s" % test_labels_file)
                np.save(test_data_file, utk_dataset_test)
                np.save(test_labels_file, utk_labels_test)

            print("Training " + model + " model")
            if model == 'CNN':
                run_keras_cnn_model(key, str(int(run_num + 1)))
            elif model == 'LSTM':
                run_keras_lstm_model(key, str(int(run_num + 1)))
            elif model == 'ConvRNN':
                run_keras_nunez_model(key, str(int(run_num + 1)))
            else:
                print("Model unknown!")
        print("Finished run #" + str(int(run_num + 1)))

print_summary()

exit(0)
