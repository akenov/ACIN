from __future__ import print_function
import keras
import pickle
import glob
import os
import numpy as np
import itertools
from datetime import datetime
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Permute
from keras.layers import Conv2D, MaxPooling2D, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from DataGenerator import DataGenerator


def normalize_data(data_):
    orig_shape = data_.shape
    data_ = np.reshape(data_, [orig_shape[0], -1])
    data_ = StandardScaler().fit_transform(data_)
    data_ = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data_)
    data_ = np.reshape(data_, orig_shape)
    return data_


def getFrameIds(filename_):
    dir_names = [
        "gesture",
        "finger",
        "subject",
        "essai"
    ]
    idxs_ = np.zeros((len(dir_names), 1), dtype=np.int64)

    num_startmark = "_"
    num_endmark = "/"
    # num_endmark = "\\"  # CHANGE ON *NIX

    for name in dir_names:
        # print(name)
        sub_pos = filename_.find(name)
        subfilename = filename_[sub_pos:]
        id_start = subfilename.find(num_startmark)
        id_end = subfilename.find(num_endmark)
        dir_num = subfilename[id_start + 1: id_end]
        idxs_[dir_names.index(name)] = dir_num
    start_frame = int(SEQ_INFO[idxs_[0]-1, idxs_[1]-1, idxs_[2]-1, idxs_[3]-1, 0])
    end_frame = int(SEQ_INFO[idxs_[0]-1, idxs_[1]-1, idxs_[2]-1, idxs_[3]-1, 1])

    return start_frame, end_frame


def append_to_set(filename, dataset, labelset, skel_data):
    # print("Temp old shape %s " % (skel_data.shape,))
    data_new = np.zeros([NUM_JOINTS, MAX_WIDTH, 3])
    for j in range(0, skel_data.shape[1], 3):
        for n in range(0, skel_data.shape[0], 1):
            # print("j=%d n=%d" % (j, n))
            dframe = np.transpose(skel_data[n, j:j + 3])
            # print("dframe shape: %s " % (dframe.shape, ))
            data_new[int(j / 3), n, :] = dframe
    # numpy has a permute - use this one for the future

    dataset.append(data_new)
    # Note: use \\ when running on Windows
    if 'gesture_1\\' in filename:
        labelset.append(LABELS[0])
        # print("Gesture: Grab")
    elif 'gesture_2' in filename:
        labelset.append(LABELS[1])
        # print("Gesture: Tap")
    elif 'gesture_3' in filename:
        labelset.append(LABELS[2])
        # print("Gesture: Expand")
    elif 'gesture_4' in filename:
        labelset.append(LABELS[3])
        # print("Gesture: Pinch")
    elif 'gesture_5' in filename:
        labelset.append(LABELS[4])
        # print("Gesture: Rotation CW")
    elif 'gesture_6' in filename:
        labelset.append(LABELS[5])
        # print("Gesture: Rotation CCW")
    elif 'gesture_7' in filename:
        labelset.append(LABELS[6])
        # print("Gesture: Swipe Right")
    elif 'gesture_8' in filename:
        labelset.append(LABELS[7])
        # print("Gesture: Swipe Left")
    elif 'gesture_9' in filename:
        labelset.append(LABELS[8])
        # print("Gesture: Swipe Up")
    elif 'gesture_10' in filename:
        labelset.append(LABELS[9])
        # print("Gesture: Swipe Down")
    elif 'gesture_11' in filename:
        labelset.append(LABELS[10])
        # print("Gesture: Swipe X")
    elif 'gesture_12' in filename:
        labelset.append(LABELS[11])
        # print("Gesture: Swipe V")
    elif 'gesture_13' in filename:
        labelset.append(LABELS[12])
        # print("Gesture: Swipe +")
    elif 'gesture_14' in filename:
        labelset.append(LABELS[13])
        # print("Gesture: Shake")
    else:
        print("ERROR! Unknown label - will cause dimensionallity mismatch.")


def load_from_fileset(fileset_):
    data_ = []
    labels_ = []
    for file in fileset_:
        # print("Processing file: %s " % file)
        skeleton_sample = np.loadtxt(file, delimiter=' ', dtype=np.float64)
        # print("Sample dimensions %s " % (skeleton_sample.shape,))
        frameid_start, frameid_end = getFrameIds(file)
        skeleton_sample = skeleton_sample[frameid_start-1: frameid_end-1, :]
        skeleton_sample = np.pad(skeleton_sample, [(0, MAX_WIDTH - skeleton_sample.shape[0]), (0, 0)], mode='constant',
                                 constant_values=0)

        append_to_set(file, data_, labels_, skeleton_sample)
    print(len(data_))
    print(len(labels_))
    return np.asarray(data_), np.asarray(labels_)


def run_keras_cnn_model(loso_, epochs_n, run_suffix, aug_list):
    modelname = 'DHG CNN LOSO ' + loso_
    batch_size_base = 5
    # regul_val = 0.015
    augmentations_ = aug_list

    train_data_file_ = "DHG.train." + loso_ + ".data.npy"
    # train_labels_file_ = "DHG.train." + loso_ + ".labels.npy"
    test_data_file_ = "DHG.test." + loso_ + ".data.npy"
    test_labels_file_ = "DHG.test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = OUTPUT_SAVES + 'dhg_mnistcnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    scoresave = OUTPUT_SAVES + 'dhg_mnistcnn_scores.' + loso_ + '.' + run_suffix + '.save'
    weightsave = OUTPUT_SAVES + 'dhg_mnistcnn_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + 'dhg_mnistcnn_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data_.shape[0], 1)
    batch_size_aug = len(augmentations_) * batch_size_base
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    # print("Input Shape = %s " % (ishape, ))

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    # Generators
    training_generator = GeneratorDHG(generator_type_train, batch_size_aug, ishape, list_idxes, augmentations_)

    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=ishape))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(NUM_CLASSES, activation='softmax'))

    cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                      #optimizer=keras.optimizers.Adadelta(),
                      optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.993, decay=0.0),
                      metrics=['accuracy'])

    cnn_model.summary()
    print(datetime.now())
    print("Start training")
    history = cnn_model.fit_generator(generator=training_generator,
                                      epochs=epochs_n,
                                      shuffle=False, use_multiprocessing=False,  # change on Linux
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


def run_keras_lstm_model(loso_, epochs_n, run_suffix, aug_list):
    modelname = 'DHG LSTM LOSO #' + loso_[4:]
    batch_size_base = 5
    regul_val = 0.015
    augmentations_ = aug_list

    train_data_file_ = "DHG.train." + loso_ + ".data.npy"
    # train_labels_file_ = "DHG.train." + loso_ + ".labels.npy"
    test_data_file_ = "DHG.test." + loso_ + ".data.npy"
    test_labels_file_ = "DHG.test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = OUTPUT_SAVES + 'dhg_lstm_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    scoresave = OUTPUT_SAVES + 'dhg_lstm_scores.' + loso_ + '.' + run_suffix + '.save'
    weightsave = OUTPUT_SAVES + 'dhg_lstm_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + 'dhg_lstm_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data_.shape[0], 1)
    batch_size_aug = len(augmentations_) * batch_size_base
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    # print("Input Shape = %s " % (ishape,))

    # Generators
    training_generator = GeneratorDHG(generator_type_train, batch_size_aug, ishape, list_idxes, augmentations_)

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    lstm_model = Sequential()
    lstm_model.add(Permute((2, 1, 3), input_shape=ishape))
    permute_shape = lstm_model.layers[0].output_shape
    resh_dim1 = permute_shape[2]
    resh_dim2 = permute_shape[1] * permute_shape[3]
    resh_shape = (resh_dim1, resh_dim2)
    lstm_model.add(Reshape(resh_shape))
    lstm_model.add(LSTM(128, input_shape=[batch_size_aug, resh_dim1, resh_dim2], stateful=False, unroll=True, kernel_regularizer=regularizers.l2(regul_val)))
    lstm_model.add(Dense(128, activation='relu'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(NUM_CLASSES, activation='softmax'))

    lstm_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adadelta(),
                       metrics=['accuracy'])

    lstm_model.summary()
    print(datetime.now())
    print("Start training")

    # history = lstm_model.fit(train_data, train_labels,
    #                          batch_size=batch_size_aug,
    #                          epochs=1, verbose=1,
    #                          validation_data=(test_data, test_labels))

    history = lstm_model.fit_generator(generator=training_generator,
                                       epochs=epochs_n,
                                       shuffle=False, use_multiprocessing=False,  # CHANGE ON RACER!
                                       callbacks=[tensorboard])

    print(datetime.now())
    print(test_data_.shape)
    scores = lstm_model.evaluate(test_data_, test_labels_)
    print(datetime.now())

    print("# KERAS MODEL: " + modelname + " # # # ")
    print('Test loss: %.4f' % scores[0])
    print('Test accuracy: %.3f %%' % (scores[1] * 100))
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


def run_keras_convrnn_model(loso_, epochs_n, run_suffix, aug_list):
    modelname = 'DHG ConvRNN LOSO #' + loso_[4:]
    batch_size_base = 5
    regul_val = 0.015
    augmentations_ = aug_list

    train_data_file_ = "DHG.train." + loso_ + ".data.npy"
    # train_labels_file_ = "DHG.train." + loso_ + ".labels.npy"
    test_data_file_ = "DHG.test." + loso_ + ".data.npy"
    test_labels_file_ = "DHG.test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = OUTPUT_SAVES + 'dhg_convrnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    weightsave = OUTPUT_SAVES + 'dhg_convrnn_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + 'dhg_convrnn_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data_.shape[0], 1)
    batch_size_aug = len(augmentations_) * batch_size_base
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    # print("Input Shape = %s " % (ishape, ))

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    # Generators
    training_generator = GeneratorDHG(generator_type_train, batch_size_aug, ishape, list_idxes, augmentations_)

    convrnn_model = Sequential()
    convrnn_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=ishape, padding='same', kernel_regularizer=regularizers.l2(regul_val)))
    convrnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    convrnn_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(regul_val)))
    convrnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    convrnn_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(regul_val)))
    convrnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    # CNN part
    # epochs_n = 100
    # model.add(Dense(300))
    # model.add(Dense(100))

    # for layer in convrnn_model.layers:
    #     print(layer.output_shape)

    # RNN part
    convrnn_model.add(Permute((2, 1, 3)))
    print(convrnn_model.layers[-1].output_shape)
    convrnn_model.add(Reshape((62, 200)))
    print(convrnn_model.layers[-1].output_shape)
    convrnn_model.add(LSTM(100, return_sequences=True, batch_input_shape=(100, 62, 200), kernel_regularizer=regularizers.l2(regul_val)))
    # model.add(Dropout(0.5))
    convrnn_model.add(Flatten())
    convrnn_model.add(Dense(NUM_CLASSES, activation='softmax'))

    convrnn_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

    convrnn_model.summary()
    print(datetime.now())
    print("Start training")
    history = convrnn_model.fit_generator(generator=training_generator,
                                          epochs=epochs_n,
                                          shuffle=False, use_multiprocessing=True,
                                          callbacks=[tensorboard])

    # history = model.fit(train_data, train_labels,
    #                     batch_size=batch_size_aug,
    #                     epochs=epochs_n, verbose=1,
    #                     validation_data=(test_data, test_labels))

    print(datetime.now())
    print(test_data_.shape)
    scores = convrnn_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug)
    print(datetime.now())

    print("# KERAS MODEL: " + modelname + " # # # ")
    print('Test loss: %.4f' % scores[0])
    print('Test accuracy: %.3f %%' % (scores[1] * 100))
    pred_labels = convrnn_model.predict(test_data_, batch_size=batch_size_aug)
    # print("Prediction matrix data:")
    # print(pred_labels.shape)
    # print(pred_labels)
    # print(datetime.now())

    with open(histsave, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        print("Saved training history %s" % histsave)

    convrnn_model.save(weightsave)
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


def run_keras_nunez_model(loso_, epochs_n, run_suffix, aug_list):
    modelname = 'DHG Nunez LOSO #' + loso_[4:]
    cnn_batch_size_base = 20
    # epochs_n_cnn = 100
    regul_val = 0.015
    augmentations_ = aug_list

    train_data_file_ = "DHG.train." + loso_ + ".data.npy"
    # train_labels_file_ = "DHG.train." + loso_ + ".labels.npy"
    test_data_file_ = "DHG.test." + loso_ + ".data.npy"
    test_labels_file_ = "DHG.test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    cnn_histsave = OUTPUT_SAVES + 'dhg_nunez_cnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    rnn_histsave = OUTPUT_SAVES + 'dhg_nunez_rnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    cnn_scoresave = OUTPUT_SAVES + 'dhg_nunez_cnn_scores.' + loso_ + '.' + run_suffix + '.save'
    rnn_weightsave = OUTPUT_SAVES + 'dhg_nunez_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + 'dhg_nunez_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    print("Test Data Shape = %s " % (test_data_.shape,))
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data_.shape[0], 1)
    batch_size_aug_cnn = len(augmentations) * cnn_batch_size_base
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    # print("Input Shape = %s " % (ishape, ))
    bi_shape = (batch_size_aug_cnn, test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    print("Batch Input Shape = %s " % (bi_shape,))

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    # Generators
    training_generator_cnn = GeneratorDHG(generator_type_train, batch_size_aug_cnn, ishape, list_idxes, augmentations_)

    conv_model = Sequential()
    conv_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=ishape, #batch_input_shape=bi_shape,
                             padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    # CNN part
    conv_model.add(Dense(300))
    conv_model.add(Dense(100))
    conv_model.add(Flatten())
    conv_model.add(Dense(NUM_CLASSES, activation='softmax'))

    conv_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.993, decay=0.0),
                       metrics=['accuracy'])

    conv_model.summary()
    print(datetime.now())
    print("Start training")
    history_cnn = conv_model.fit_generator(generator=training_generator_cnn,
                                           epochs=epochs_n,
                                           shuffle=False, use_multiprocessing=True,
                                           validation_data=(test_data_, test_labels_),
                                           callbacks=[tensorboard])

    cnn_scores = conv_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug_cnn)
    print(datetime.now())

    with open(cnn_scoresave, 'wb') as file_pi:
        pickle.dump(cnn_scores, file_pi)
        print("Saved training history %s" % cnn_scoresave)

    with open(cnn_histsave, 'wb') as file_pi:
        pickle.dump(history_cnn.history, file_pi)
        print("Saved training history %s" % cnn_histsave)

    conv_model.layers.pop()  # Dense(Softmax)
    conv_model.layers.pop()  # Flatten()
    conv_model.layers.pop()  # Dense(100)
    conv_model.layers.pop()  # Dense(300)
    print("Continue with RNN")

    # RNN part
    rnn_batch_size_base = 6
    batch_size_aug_rnn = len(augmentations) * rnn_batch_size_base

    training_generator_rnn = GeneratorDHG(generator_type_train, batch_size_aug_rnn, ishape, list_idxes, augmentations_)
    # for layer in convrnn_model.layers:
    #     print(layer.output_shape)

    convrnn_model = Sequential()
    convrnn_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=ishape, #batch_input_shape=bi_shape,
                             padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    convrnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    convrnn_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    convrnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    convrnn_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    convrnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    convrnn_model.set_weights(conv_model.get_weights())

    convrnn_model.add(Permute((2, 1, 3)))
    #print(convrnn_model.layers[-1].output_shape)
    convrnn_model.add(Reshape((62, 200)))
    #print(convrnn_model.layers[-1].output_shape)
    convrnn_model.add(LSTM(100, return_sequences=True, stateful=False, unroll=True))  #batch_input_shape=(100, 25, 200), , kernel_regularizer=regularizers.l2(regul_val)
    convrnn_model.add(Flatten())
    convrnn_model.add(Dense(NUM_CLASSES, activation='softmax'))

    convrnn_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.993, decay=0.0),
                          metrics=['accuracy'])

    convrnn_model.summary()
    print(datetime.now())
    print("Start training")
    history_rnn = convrnn_model.fit_generator(generator=training_generator_rnn,
                                              epochs=int(epochs_n*5),
                                              shuffle=False, use_multiprocessing=True,
                                              validation_data=(test_data_, test_labels_),
                                              callbacks=[tensorboard])

    print(datetime.now())
    # print(test_data_.shape)
    scores = convrnn_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug_cnn)
    print(datetime.now())

    print("# KERAS MODEL: " + modelname + " # # # ")
    print('Test loss: %.4f' % scores[0])
    print('Test accuracy: %.3f %%' % (scores[1] * 100))
    pred_labels = convrnn_model.predict(test_data_, batch_size=batch_size_aug_cnn)
    # print("Prediction matrix data:")
    # print(pred_labels.shape)
    # print(pred_labels)
    # print(datetime.now())

    with open(rnn_histsave, 'wb') as file_pi:
        pickle.dump(history_rnn.history, file_pi)
        print("Saved RNN training history %s" % rnn_histsave)

    convrnn_model.save(rnn_weightsave)
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
    ('kfold0', 'subject_1/'),
    ('kfold0', 'subject_2/'),
    ('kfold0', 'subject_3/'),
    ('kfold0', 'subject_4/'),
    ('kfold2', 'subject_5/'),
    ('kfold2', 'subject_6/'),
    ('kfold2', 'subject_7/'),
    ('kfold2', 'subject_8/'),
    ('kfold4', 'subject_9/'),
    ('kfold4', 'subject_10/'),
    ('kfold4', 'subject_11/'),
    ('kfold4', 'subject_12/'),
    ('kfold6', 'subject_13/'),
    ('kfold6', 'subject_14/'),
    ('kfold6', 'subject_15/'),
    ('kfold6', 'subject_16/'),
    ('kfold8', 'subject_17/'),
    ('kfold8', 'subject_18/'),
    ('kfold8', 'subject_19/'),
    ('kfold8', 'subject_20/')
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

NUM_CLASSES = 14
MAX_WIDTH = 500
NUM_JOINTS = 22

# PARAMETERS #
iterations = 3
num_epochs = 1
DHGFOLDER = "D:\\!DA-20092018\\DHG2016/"
# DHGFOLDER = "./DHG2016/"
allfiles_list = glob.glob(DHGFOLDER + "/*/finger_1/*/*/skeleton_world.txt")
print("DHG14 - working only with FINGER 1")
info_list_file = "./DHG2016/informations_troncage_sequences.txt"
# SET OUTPUT_SAVES OUTSIDE THE DOCKER CONTAINER
OUTPUT_SAVES = "/DHG/saves/"
# AUGMENTATIONS: none, shift, scale, noise, subsample, interpol
augmentations = [
    'none',
    # 'shift',
    # 'scale',
    # 'noise',
    # 'subsample',
    # 'interpol'
]
# MODELS: CNN, LSTM, ConvRNN
train_models = [
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

for model in train_models:
    for run in np.arange(0, iterations, 1):
        for key, batch_group in itertools.groupby(batch_names, lambda x: x[0]):
            # for subject in subject_list:
            # loso_id = subject.split('_')[1]
            # loso = "loso" + loso_id[:len(loso_id)-1]
            # loso = subject[:len(subject)-1].replace('_', '').upper()
            print("Leaving out batch " + key)
            print("Augmentations: %s" % augmentations)

            test_files = []
            for batch_pair in batch_group:
                for testfile in glob.glob(DHGFOLDER + "/*/finger_1/" + batch_pair[1] + "/*/skeleton_world.txt"):
                    test_files.append(testfile)
            train_files = allfiles_list[:]
            for testfile in test_files:
                train_files.remove(testfile)

            train_data_file = "DHG.train." + key + ".data"
            train_labels_file = "DHG.train." + key + ".labels"
            test_data_file = "DHG.test." + key + ".data"
            test_labels_file = "DHG.test." + key + ".labels"

            if os.path.exists(train_data_file + ".npy") and os.path.exists(train_labels_file + ".npy") \
                    and os.path.exists(test_data_file + ".npy") and os.path.exists(test_labels_file + ".npy"):
                print("Sample TRAIN and TEST data sets files found. Skipping generation from skeleton data.")
            else:
                print("Generating TRAIN data set. May take few minutes...")
                train_data, train_labels = load_from_fileset(train_files)
                print("Generating TEST data set. May take a minute...")
                test_data, test_labels = load_from_fileset(test_files)

                train_data = normalize_data(train_data)
                test_data = normalize_data(test_data)

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
                run_keras_cnn_model(key, num_epochs, str(int(run + 1)), augmentations)
            elif model == 'LSTM':
                run_keras_lstm_model(key, num_epochs, str(int(run + 1)), augmentations)
            elif model == 'ConvRNN':
                run_keras_nunez_model(key, num_epochs, str(int(run + 1)), augmentations)
            else:
                print("Model unknown!")

        print("Finished run #" + str(int(run + 1)))

exit(0)
