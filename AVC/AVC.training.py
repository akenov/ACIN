from __future__ import print_function

import pandas as pd
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


def sliding_window_generator(sequence, label):
    label = label.reshape([1, -1])
    slided_window_samples = np.zeros([1, sequence.shape[0], sequence.shape[1], sequence.shape[2]])
    slided_window_labels = np.zeros([1, label.shape[1]])

    clean_width = int(sequence[~np.all(sequence == 0.0, axis=2)].shape[0]/sequence.shape[1])
    sequence_clean = sequence[~np.all(sequence == 0.0, axis=2)] \
        .reshape([clean_width, sequence.shape[1], sequence.shape[2]])

    for frame_id in range(sequence_clean.shape[0] - SLIDINGWINDOW_SIZE + 1):
        window = sequence_clean[frame_id: frame_id+SLIDINGWINDOW_SIZE, :, :]
        sequence_padded = np.pad(window, [(0, sequence.shape[0] - SLIDINGWINDOW_SIZE), (0, 0), (0, 0)],
                                 mode='constant', constant_values=0)
        slided_window_samples = np.append(slided_window_samples, sequence_padded.reshape(
            [1, sequence.shape[0], sequence.shape[1], sequence.shape[2]]), axis=0)
        slided_window_labels = np.append(slided_window_labels, label, axis=0)

    return slided_window_samples[1:, :, :, :], slided_window_labels[1:, :]


def load_from_file(list_of_files):
    fshape = [MAX_WIDTH, NUM_JOINTS, 3]
    fdapool = []
    feapool = []
    place_present = False

    for file in list_of_files:
        print("Loading experiment: " + file)
        # fdata = pd.read_csv(file, sep=",", header=0, usecols=CLMNS_JOINTS).as_matrix()
        # flabel = pd.read_csv(file, sep=",", header=0, usecols=CLMNS_LABEL_FINE).as_matrix()

        df = pd.read_csv(file, sep=",", header=0, usecols=CLMNS_JOINTS)
        fdata = df.values
        dl = pd.read_csv(file, sep=",", header=0, usecols=CLMNS_LABEL_FINE)
        flabel = dl.values
        dtch = pd.read_csv(file, sep=",", header=0, usecols=CLMNS_TOUCH)

        ftouch = dtch.values
        ftouch = np.append(ftouch, np.zeros([ftouch.shape[0], 1]), axis=1)
        fdata = np.append(fdata, ftouch, axis=1)

        # Subselect data corresponding to the labels of interest
        # max_len = 0 # read 229, thus 500 fits all augmentation schemes
        place_present = False
        for label in VALID_LABELS:
            sample_ = np.zeros(fshape)
            s_idx = 0
            label_series = False
            for i in np.arange(len(flabel)-1):
                if flabel[i] == label:
                    label_series = True
                    row = np.array(fdata[i]).reshape(NUM_JOINTS, 3)
                    sample_[s_idx, :, :] = row
                    s_idx += 1
                if i < len(flabel) and flabel[i + 1] != label and label_series:
                    # print("LABEL JUMP DETECTED: %s " % flabel[i+1])
                    label_series = False
                    print("Orignal sample for action: %s with length %d" % (label, s_idx))
                    if 100 < s_idx <= 200:
                        substep = 2
                    elif 200 < s_idx <= 300:
                        substep = 3
                    elif 300 < s_idx <= 400:
                        substep = 4
                    elif s_idx > 400:
                        substep = 5
                    else:
                        substep = 1
                    subrange = np.arange(0, s_idx, substep)
                    print("Adding subsample with length %d " % len(subrange))
                    sub_sample = np.pad(sample_[subrange], [(0, MAX_WIDTH - len(subrange)), (0, 0), (0, 0)],
                                        mode='constant', constant_values=0)
                    fdapool.append(sub_sample)
                    feapool.append(label)
                    sample_ = np.zeros(fshape)
                    s_idx = 0
                if place_present is False and flabel[i] == 'place':
                    place_present = True

    # Fix OH Encoding for PLACE not present in all experiments
    if place_present is False:
        fdapool.append(np.zeros(fshape))
        feapool.append('place')

    data_raw = np.reshape(fdapool, [-1, MAX_WIDTH, NUM_JOINTS, 3])

    # print(le.fit(feapool).classes_)
    # transforms alphabetically: A->Z : 0->25
    # G..rab -> 0
    # M..oveObj -> 1
    # P..lace -> 2
    # R..each -> 3
    # Generate numeric feature vector via LabelEncoder()
    feat_labenc = le.fit_transform(np.array(feapool))
    # Generate OneHot feature matrix via OneHotEncoder()
    # feat_onehot = ohe.fit_transform(feat_labenc.reshape(len(feat_labenc), 1))
    feat_onehot = ohe.fit_transform(np.array(feapool).reshape(len(feat_labenc), 1))
    print("Final dataset dimensions: " + str(data_raw.shape))
    return data_raw, feat_onehot


def gen_file_set(expr_set):
    fileset = []
    if len(expr_set) == 2:
        return EXPERIMENTS_DIR + "/AnnotationExperiment" + expr_set[1] + "Merged.csv"
    for num in expr_set:
        fileset.append(EXPERIMENTS_DIR + "/AnnotationExperiment" + num[1] + "Merged.csv")
    return fileset


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

    # cnn_model.add(BatchNormalization(input_shape=ishape))
    # cnn_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False))
    # cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # cnn_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same', use_bias=False))
    # cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # cnn_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False))
    # cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # cnn_model.add(Dropout(COEFF_DROPOUT))
    # cnn_model.add(Dense(300, activation='relu', use_bias=False))
    # cnn_model.add(Dropout(COEFF_DROPOUT))
    # cnn_model.add(Dense(100, activation='relu', use_bias=False))
    # cnn_model.add(Flatten())
    # cnn_model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Old MNIST Model with single Dense layer
    cnn_model.add(BatchNormalization(input_shape=ishape))
    cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', use_bias=False))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', use_bias=False))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(COEFF_DROPOUT))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu', use_bias=False))
    cnn_model.add(Dropout(COEFF_DROPOUT))
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
    lstm_model.add(LSTM(128, return_sequences=True, stateful=False))
    lstm_model.add(LSTM(64, stateful=False))
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
                                       epochs=4*NUM_EPOCHS, validation_data=(test_data_, test_labels_),
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
    nunez_model.add(Reshape((62, 200)))
    # print(nunez_model.layers[-1].output_shape)
    nunez_model.add(Masking(mask_value=0.0, input_shape=nunez_model.layers[-1].output_shape))
    nunez_model.add(BatchNormalization())
    nunez_model.add(LSTM(100, kernel_regularizer=regularizers.l2(COEFF_REGULARIZATION_L2),
                         stateful=False, use_bias=False))
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
    # print("| EXTEND_ACTIONS: " + str(EXTEND_ACTIONS))
    print("| USE_SLIDINGWINDOW: " + str(USE_SLIDINGWINDOW))
    # print("| USE_SCALER: " + str(USE_SCALER))
    print("| CNN_TRAINABLE: " + str(CNN_TRAINABLE))
    print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")
    print("| OPTIMIZER: " + OPTIMIZER[0])
    print("| CNN_BATCH_SIZE: " + str(CNN_BATCH_SIZE))
    print("| RNN_BATCH_SIZE: " + str(RNN_BATCH_SIZE))
    print("| FRAMES_THRESHOLD: " + str(FRAMES_THRESHOLD))
    print("| SLIDINGWINDOW_SIZE: " + str(SLIDINGWINDOW_SIZE))
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


CLMNS_JOINTS = [
    'boneCenter000X',
    'boneCenter000Z',
    'boneCenter000Y',

    'boneCenter001X',
    'boneCenter001Z',
    'boneCenter001Y',

    'boneCenter002X',
    'boneCenter002Z',
    'boneCenter002Y',

    'boneCenter003X',
    'boneCenter003Z',
    'boneCenter003Y',

    'boneCenter010X',
    'boneCenter010Z',
    'boneCenter010Y',

    'boneCenter011X',
    'boneCenter011Z',
    'boneCenter011Y',

    'boneCenter012X',
    'boneCenter012Z',
    'boneCenter012Y',

    'boneCenter013X',
    'boneCenter013Z',
    'boneCenter013Y',

    'boneCenter020X',
    'boneCenter020Z',
    'boneCenter020Y',

    'boneCenter021X',
    'boneCenter021Z',
    'boneCenter021Y',

    'boneCenter022X',
    'boneCenter022Z',
    'boneCenter022Y',

    'boneCenter023X',
    'boneCenter023Z',
    'boneCenter023Y',

    'boneCenter030X',
    'boneCenter030Z',
    'boneCenter030Y',

    'boneCenter031X',
    'boneCenter031Z',
    'boneCenter031Y',

    'boneCenter032X',
    'boneCenter032Z',
    'boneCenter032Y',

    'boneCenter033X',
    'boneCenter033Z',
    'boneCenter033Y',

    'boneCenter040X',
    'boneCenter040Z',
    'boneCenter040Y',

    'boneCenter041X',
    'boneCenter041Z',
    'boneCenter041Y',

    'boneCenter042X',
    'boneCenter042Z',
    'boneCenter042Y',

    'boneCenter043X',
    'boneCenter043Z',
    'boneCenter043Y',

    'wristPosition0X',
    'wristPosition0Z',
    'wristPosition0Y',

    'elbowPosition0X',
    'elbowPosition0Z',
    'elbowPosition0Y',
]

CLMNS_LABEL_FINE = [
    'fineAnnotation'
]

CLMNS_TOUCH = [
    "PringlesTouchesHand",
    "TeaboxTouchesHand",
]

EXPR_SET = [
     1,  2,  3,  4,  5,  6,  7,  8,  9,
    11, 12, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51
]

batch_names = [
    ('kfold0', '48'),
    ('kfold0', '49'),
    ('kfold0', '50'),
    ('kfold0', '51'),
    ('kfold2', '1'),
    ('kfold2', '2'),
    ('kfold2', '3'),
    ('kfold2', '4'),
    ('kfold2', '5'),
    ('kfold2', '6'),
    ('kfold2', '7'),
    ('kfold2', '8'),
    ('kfold2', '9'),
    ('kfold2', '11'),
    ('kfold2', '12'),
    ('kfold2', '18'),
    ('kfold2', '19'),
    ('kfold2', '20'),
    ('kfold2', '21'),
    ('kfold2', '22'),
    ('kfold2', '23'),
    ('kfold2', '24'),
    ('kfold2', '25'),
    ('kfold2', '26'),
    ('kfold2', '28'),
    ('kfold2', '29'),
    ('kfold2', '30'),
    ('kfold2', '31'),
    ('kfold2', '32'),
    ('kfold2', '33'),
    ('kfold2', '34'),
    ('kfold2', '35'),
    ('kfold2', '36'),
    ('kfold2', '37'),
    ('kfold2', '38'),
    ('kfold2', '39'),
    ('kfold2', '40'),
    ('kfold2', '42'),
    ('kfold2', '43'),
    ('kfold2', '44'),
    ('kfold2', '45'),
    ('kfold2', '46'),
    ('kfold2', '47'),
]

VALID_LABELS = ["reach", "grab", "moveObject", "place"]

DATASET_NAME = 'AVCExt'
NUM_CLASSES = 4
MAX_WIDTH = 500
NUM_JOINTS = 23 # 22 hand joints & 1 including touch info
# PARAMETERS #

# EXPERIMENTS_DIR = "./AVCexperimentsData/"
# MULTI_CPU = True
EXPERIMENTS_DIR = "D:\\!DA-20092018\\AVCexperimentsData"
MULTI_CPU = False
# SET OUTPUT_SAVES OUTSIDE THE DOCKER CONTAINER
OUTPUT_SAVES = "./"
# EXTEND_ACTIONS = True
USE_SLIDINGWINDOW = True
# USE_SCALER = False
CNN_TRAINABLE = True
FRAMES_THRESHOLD = 13
SLIDINGWINDOW_SIZE = 80
COEFF_DROPOUT = 0.6
COEFF_REGULARIZATION_L2 = 0.015
CNN_BATCH_SIZE = 50
RNN_BATCH_SIZE = 16
k.set_epsilon(1e-06)

ITERATIONS = 1
NUM_EPOCHS = 1
AUGMENTATIONS = [
    'none',
    # "scale_shift",
    # 'scale',
    # 'shift_gauss_xy',
    # 'noise',
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

sample_filelist = gen_file_set(batch_names)

RESULTS_DIR = OUTPUT_SAVES + DATASET_NAME + "." + datetime.today().strftime('%d-%m-%Y_%H%M') + "/"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

for model in TRAIN_MODELS:
    for run in np.arange(0, ITERATIONS, 1):
        RESULTS = []
        CNN_RESULTS = []
        for key, batch_group in itertools.groupby(batch_names, lambda x: x[0]):
            print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")
            print("| File Batch: " + key)
            print("| Augmentations: %s" % AUGMENTATIONS)
            print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")

            #     #     #      #      #      #      #      #      #      #      #
            # SPEED UP RELATIVE EVAL: @JB,@MATT - USE SINGLE SPLIT
            if key != "kfold0":
                print("Skipping split " + key)
                continue
            #     #     #      #      #      #      #      #      #      #      #

            # if EXTEND_ACTIONS:
            #     print("Extend Actions #Frames Threshold: %d" % FRAMES_THRESHOLD)
            # if USE_SLIDINGWINDOW:
            #     print("Sliding Window Length: %.2f" % COEFF_SLIDINGWINDOW)

            test_files = []
            for batch_pair in batch_group:
                test_files.append(gen_file_set(batch_pair))
            train_files = sample_filelist[:]
            for testfile in test_files:
                train_files.remove(testfile)

            train_data_file = DATASET_NAME + ".train." + key + ".data"
            train_labels_file = DATASET_NAME + ".train." + key + ".labels"
            test_data_file = DATASET_NAME + ".test." + key + ".data"
            test_labels_file = DATASET_NAME + ".test." + key + ".labels"

            if os.path.exists(train_data_file + ".npy") and os.path.exists(train_labels_file + ".npy") \
                    and os.path.exists(test_data_file + ".npy") and os.path.exists(test_labels_file + ".npy"):
                print("Sample TRAIN and TEST data sets files found. Skipping generation from skeleton data.")
            else:
                train_data, train_labels = load_from_file(train_files)
                test_data, test_labels = load_from_file(test_files)

                if USE_SLIDINGWINDOW:
                    print("Generating sliding windows samples. This can take a while...")
                    for sample_id in range(train_data.shape[0]):
                        train_data_sliwin, train_labels_sliwin = sliding_window_generator(
                            train_data[sample_id, :, :, :], train_labels[sample_id, :])
                        train_data = np.append(train_data, train_data_sliwin, axis=0)
                        train_labels = np.append(train_labels, train_labels_sliwin, axis=0)

                train_data_fin, train_labels_fin = shuffle(train_data, train_labels, random_state=42)
                test_data_fin, test_labels_fin = shuffle(test_data, test_labels, random_state=42)

                print("Saving data & labels to files: ")
                print("TRAIN data  : %s" % train_data_file)
                print("TRAIN labels: %s" % train_labels_file)
                np.save(train_data_file, train_data_fin)
                np.save(train_labels_file, train_labels_fin)
                print("TEST data  : %s" % test_data_file)
                print("TEST labels: %s" % test_labels_file)
                np.save(test_data_file, test_data_fin)
                np.save(test_labels_file, test_labels_fin)

            print("Training " + model + " model")
            if model == 'CNN':
                run_keras_cnn_model(key, str(int(run + 1)))
            elif model == 'LSTM':
                run_keras_lstm_model(key, str(int(run + 1)))
            elif model == 'ConvRNN':
                run_keras_nunez_model(key, str(int(run + 1)))
            else:
                print("Model unknown!")
        print("Finished run #" + str(int(run + 1)))

print_summary()

exit(0)
