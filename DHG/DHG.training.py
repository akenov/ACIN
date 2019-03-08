from __future__ import print_function
import keras
import pickle
import math
import glob
import os
import numpy as np
import itertools
from datetime import datetime
from keras import backend as k
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Permute
from keras.layers import Conv2D, MaxPooling2D, LSTM, Masking, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from DataGenerator import DataGenerator


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


def get_label_id(filename):
    if 'gesture_1'+FLD_SLSH in filename:
        # print("Gesture: Grab")
        return 0
    elif 'gesture_2'+FLD_SLSH in filename:
        # print("Gesture: Tap")
        return 1
    elif 'gesture_3'+FLD_SLSH in filename:
        # print("Gesture: Expand")
        return 2
    elif 'gesture_4'+FLD_SLSH in filename:
        # print("Gesture: Pinch")
        return 3
    elif 'gesture_5'+FLD_SLSH in filename:
        # print("Gesture: Rotation CW")
        return 4
    elif 'gesture_6'+FLD_SLSH in filename:
        # print("Gesture: Rotation CCW")
        return 5
    elif 'gesture_7'+FLD_SLSH in filename:
        # print("Gesture: Swipe Right")
        return 6
    elif 'gesture_8'+FLD_SLSH in filename:
        # print("Gesture: Swipe Left")
        return 7
    elif 'gesture_9'+FLD_SLSH in filename:
        # print("Gesture: Swipe Up")
        return 8
    elif 'gesture_10'+FLD_SLSH in filename:
        # print("Gesture: Swipe Down")
        return 9
    elif 'gesture_11'+FLD_SLSH in filename:
        # print("Gesture: Swipe X")
        return 10
    elif 'gesture_12'+FLD_SLSH in filename:
        # print("Gesture: Swipe V")
        return 11
    elif 'gesture_13'+FLD_SLSH in filename:
        # print("Gesture: Swipe +")
        return 12
    elif 'gesture_14'+FLD_SLSH in filename:
        # print("Gesture: Shake")
        return 13
    else:
        print("ERROR! Unknown label - will cause dimensionallity mismatch.")


def sliding_window_generator(sample):
    window_size = math.floor(COEFF_SLIDINGWINDOW * sample.shape[0])
    slided_window_samples = []
    for frame in range(sample.shape[0] - window_size + 1):
        window = sample[frame: frame+window_size, :]
        slided_window_samples.append(window)
    return slided_window_samples


def get_frame_ids(filename_):
    dir_names = [
        "gesture",
        "finger",
        "subject",
        "essai"
    ]
    idxs_ = np.zeros((len(dir_names), 1), dtype=np.int64)

    for name in dir_names:
        # print(name)
        sub_pos = filename_.find(name)
        subfilename = filename_[sub_pos:]
        id_start = subfilename.find(STARTMARK)
        id_end = subfilename.find(FLD_SLSH)
        dir_num = subfilename[id_start + 1: id_end]
        idxs_[dir_names.index(name)] = dir_num
    start_frame = int(SEQ_INFO[idxs_[0]-1, idxs_[1]-1, idxs_[2]-1, idxs_[3]-1, 0])
    end_frame = int(SEQ_INFO[idxs_[0]-1, idxs_[1]-1, idxs_[2]-1, idxs_[3]-1, 1])

    return start_frame, end_frame


def append_file_to_set(filename, dataset, labelset, skel_data):
    if EXTEND_ACTIONS:
        skel_data = extend_sequences(skel_data, FRAMES_THRESHOLD)
    label = LABELS[get_label_id(filename)]

    dataset.append(skeleton_reshape(skel_data))
    labelset.append(label)

    if USE_SLIDINGWINDOW:
        for windowed_sample in sliding_window_generator(skel_data):
            dataset.append(skeleton_reshape(windowed_sample))
            labelset.append(label)


def load_from_fileset(fileset_):
    data_ = []
    labels_ = []
    for file in fileset_:
        print("Processing file: %s " % file)
        skeleton_sample = np.loadtxt(file, delimiter=' ', dtype=np.float64)
        frameid_start, frameid_end = get_frame_ids(file)
        # print("frameid_start = %d" % frameid_start)
        # print("frameid_end = %d" % frameid_end)
        if frameid_start == 0:  # Inconsistent mapping with idx 0 by map file
            frameid_start += 1
            # print("frameid_start = %d" % frameid_start)
        skeleton_sample = skeleton_sample[frameid_start-1: frameid_end-1, :]
        print("Sample length %d " % skeleton_sample.shape[0])
        append_file_to_set(file, data_, labels_, skeleton_sample)
    # print(len(data_))
    # print(len(labels_))
    return np.asarray(data_), np.asarray(labels_)


def run_keras_cnn_model(loso_, run_suffix):
    modelname = 'DHG CNN LOSO ' + loso_

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = RESULTS_DIR + DATASET_NAME + '.mnistcnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    scoresave = RESULTS_DIR + DATASET_NAME + '.mnistcnn_scores.' + loso_ + '.' + run_suffix + '.save'
    weightsave = RESULTS_DIR + DATASET_NAME + '.mnistcnn_weights.' + loso_ + '.' + run_suffix + '.h5'
    cfsave = RESULTS_DIR + DATASET_NAME + '.mnistcnn_confusion_matrix.' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(AUGMENTATIONS) * train_data_.shape[0], 1)
    batch_size_aug = len(AUGMENTATIONS) * CNN_BATCH_SIZE
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    print("Input Shape = %s " % (ishape, ))

    tensorboard = TensorBoard(log_dir=RESULTS_DIR, histogram_freq=0,
                              write_graph=True, write_images=True)
    training_generator = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug,
                                       ishape, list_idxes, AUGMENTATIONS)

    cnn_model = Sequential()

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
    modelname = 'DHG LSTM LOSO #' + loso_[4:]

    train_data_file_ = "DHG.train." + loso_ + ".data.npy"
    # train_labels_file_ = "DHG.train." + loso_ + ".labels.npy"
    test_data_file_ = "DHG.test." + loso_ + ".data.npy"
    test_labels_file_ = "DHG.test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = RESULTS_DIR + 'dhg_lstm_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    scoresave = RESULTS_DIR + 'dhg_lstm_scores.' + loso_ + '.' + run_suffix + '.save'
    weightsave = RESULTS_DIR + 'dhg_lstm_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = RESULTS_DIR + 'dhg_lstm_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(AUGMENTATIONS) * train_data_.shape[0], 1)
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
                                       epochs=4*NUM_EPOCHS,
                                       shuffle=False, use_multiprocessing=MULTI_CPU,
                                       callbacks=[tensorboard])

    print(datetime.now())
    print(test_data_.shape)
    scores = lstm_model.evaluate(test_data_, test_labels_)
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
    print("Train Data Shape = %s " % (train_data_.shape,))
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
    nunez_model.add(Reshape((18, 200)))
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
                                            epochs=NUM_EPOCHS*5, validation_data=(test_data_, test_labels_),
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
    print("| EXTEND_ACTIONS: " + str(EXTEND_ACTIONS))
    print("| USE_SLIDINGWINDOW: " + str(USE_SLIDINGWINDOW))
    print("| USE_SCALER: " + str(USE_SCALER))
    print("| CNN_TRAINABLE: " + str(CNN_TRAINABLE))
    print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")
    print("| OPTIMIZER: " + OPTIMIZER[0])
    print("| NUM_EPOCHS: " + NUM_EPOCHS)
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


subject_list = [
    'subject_1/',
    'subject_2/',
    'subject_3/',
    'subject_4/',
    'subject_5/',
    'subject_6/',
    'subject_7/',
    'subject_8/',
    'subject_9/',
    'subject_10/',
    'subject_11/',
    'subject_12/',
    'subject_13/',
    'subject_14/',
    'subject_15/',
    'subject_16/',
    'subject_17/',
    'subject_18/',
    'subject_19/',
    'subject_20/'
]

batch_names = [
    ('kfold2', 'subject_1/'),
    ('kfold2', 'subject_2/'),
    ('kfold2', 'subject_3/'),
    ('kfold2', 'subject_4/'),
    ('kfold2', 'subject_5/'),
    ('kfold2', 'subject_6/'),
    ('kfold0', 'subject_7/'),
    ('kfold0', 'subject_8/'),
    ('kfold2', 'subject_9/'),
    ('kfold2', 'subject_10/'),
    ('kfold2', 'subject_11/'),
    ('kfold2', 'subject_12/'),
    ('kfold2', 'subject_13/'),
    ('kfold2', 'subject_14/'),
    ('kfold2', 'subject_15/'),
    ('kfold2', 'subject_16/'),
    ('kfold2', 'subject_17/'),
    ('kfold2', 'subject_18/'),
    ('kfold2', 'subject_19/'),
    ('kfold2', 'subject_20/')
]

LABELS = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]

DATASET_NAME = 'DHG'
NUM_CLASSES = 14
MAX_WIDTH = 150
NUM_JOINTS = 22
STARTMARK = "_"
DHG_FINGER = "2"
# PARAMETERS #
# FLD_SLSH = '/'  # USE for *NIX
# DHGFOLDER = "./DHG2016/"
# MULTI_CPU = True

FLD_SLSH = '\\'  # USE for Windows
DHGFOLDER = "D:\\!DA-20092018\\DHG2016/"
MULTI_CPU = False

allfiles_list = glob.glob(DHGFOLDER + "/*/finger_" + DHG_FINGER + "/*/*/skeleton_world.txt")
print("DHG14 - working only with FINGER " + DHG_FINGER)
info_list_file = DHGFOLDER + "/informations_troncage_sequences.txt"
# SET OUTPUT_SAVES OUTSIDE THE DOCKER CONTAINER
OUTPUT_SAVES = "./"
EXTEND_ACTIONS = True
USE_SLIDINGWINDOW = True
USE_SCALER = False
CNN_TRAINABLE = True
FRAMES_THRESHOLD = 13
COEFF_SLIDINGWINDOW = 0.8
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

frames_info = np.loadtxt(info_list_file, dtype=np.uint32, delimiter=" ")
# 14 Gestures, 2 Fingers, 20 Subjects, 5 Tries, 2 Values
SEQ_INFO = np.zeros([14, 2, 20, 5, 2], dtype=np.int64)
for nseq in np.arange(0, len(frames_info), 1):
    row = frames_info[nseq, :]
    SEQ_INFO[row[0]-1, row[1]-1, row[2]-1, row[3]-1, 0] = row[4]
    SEQ_INFO[row[0]-1, row[1]-1, row[2]-1, row[3]-1, 1] = row[5]

RESULTS_DIR = OUTPUT_SAVES + DATASET_NAME + "." + datetime.today().strftime('%d-%m-%Y_%H%M') + "/"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

for model in TRAIN_MODELS:
    for run in np.arange(0, ITERATIONS, 1):
        RESULTS = []
        CNN_RESULTS = []
        for key, batch_group in itertools.groupby(batch_names, lambda x: x[0]):
            # for subject in subject_list:
            # loso_id = subject.split('_')[1]
            # loso = "loso" + loso_id[:len(loso_id)-1]
            # loso = subject[:len(subject)-1].replace('_', '').upper()
            print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")
            print("| File Batch: " + key)
            print("| Augmentations: %s" % AUGMENTATIONS)
            if EXTEND_ACTIONS:
                print("| Extend Actions #Frames Threshold: %d" % FRAMES_THRESHOLD)
            if USE_SLIDINGWINDOW:
                print("| Sliding Window Length: %.2f" % COEFF_SLIDINGWINDOW)
            print("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")

            #     #     #      #      #      #      #      #      #      #      #
            # SPEED UP RELATIVE EVAL: @JB,@MATT - USE SINGLE SPLIT
            if key != "kfold0":
                print("Skipping split " + key)
                continue
            #     #     #      #      #      #      #      #      #      #      #

            test_files = []
            for batch_pair in batch_group:
                for testfile in glob.glob(DHGFOLDER + "/*/finger_" + DHG_FINGER + "/"
                                          + batch_pair[1] + "/*/skeleton_world.txt"):
                    test_files.append(testfile)
            train_files = allfiles_list[:]
            for testfile in test_files:
                train_files.remove(testfile)
            print(test_files)
            print("- - - - - - - - - - - -")
            print(train_files)

            train_data_file = DATASET_NAME + ".train." + key + ".data"
            train_labels_file = DATASET_NAME + ".train." + key + ".labels"
            test_data_file = DATASET_NAME + ".test." + key + ".data"
            test_labels_file = DATASET_NAME + ".test." + key + ".labels"

            if os.path.exists(train_data_file + ".npy") and os.path.exists(train_labels_file + ".npy") \
                    and os.path.exists(test_data_file + ".npy") and os.path.exists(test_labels_file + ".npy"):
                print("Sample TRAIN and TEST data sets files found. Skipping generation from skeleton data.")
            else:
                print("Generating TRAIN data set. May take few minutes...")
                train_data, train_labels = load_from_fileset(train_files)
                print("Generating TEST data set. May take a minute...")
                test_data, test_labels = load_from_fileset(test_files)

                # train_data = normalize_data(train_data)
                # test_data = normalize_data(test_data)

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
