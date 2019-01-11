from __future__ import print_function
import keras
import pickle
import os
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Permute
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from DataGenerator import DataGenerator


def normalize_data(data_):
    orig_shape = data_.shape
    data_ = np.reshape(data_, [orig_shape[0], -1])
    data_ = StandardScaler().fit_transform(data_)
    data_ = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data_)
    data_ = np.reshape(data_, orig_shape)
    return data_


def load_from_file(list_of_files):
    fshape = [22, 500, 3]
    fdapool = []
    feapool = []
    place_present = False

    for file in list_of_files:
        print("Loading experiment: " + file)
        # load bones coordinates
        fdata = pd.read_csv(file, sep=" ", header=0, usecols=CLMNS_JOINTS).as_matrix()
        # print(fdata.shape)
        flabel = pd.read_csv(file, sep=" ", header=0, usecols=CLMNS_LABL_FIN).as_matrix()

        # Subselect data corresponding to the labels of interest
        # max_len = 0 # read 229, thus 500 fits all augmentation schemes
        place_present = False
        for label in VALID_LABELS:
            sample_ = np.zeros(fshape)
            s_idx = 0
            label_series = False
            for i in np.arange(0, len(flabel)-1, 1):
                if flabel[i] == label:
                    label_series = True
                    row = np.array(fdata[i]).reshape(22, 3)
                    sample_[:, s_idx, :] = row
                    s_idx += 1
                if i < len(flabel) and flabel[i + 1] != label and label_series:
                    # print("LABEL JUMP DETECTED: %s " % flabel[i+1])
                    label_series = False
                    # print("Saving sample for action: %s with length %d" % (label, s_idx))
                    fdapool.append(sample_)
                    feapool.append(label)
                    sample_ = np.zeros(fshape)
                    # if s_idx > max_len:
                    #     max_len = s_idx
                    s_idx = 0
                if place_present is False and flabel[i] == 'place':
                    place_present = True

    # Fix OH Encoding for PLACE not present in all experiments
    if place_present is False:
        fdapool.append(np.zeros(fshape))
        feapool.append('place')

    data_raw = np.reshape(fdapool, [-1, 22, 500, 3])
    # https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
    # https://stackoverflow.com/questions/42460217/how-to-normalize-a-4d-numpy-array
    # data = normalize_data(data_raw)

    print(le.fit(feapool).classes_)
    # transforms alphabetically: A->Z : 0->25
    # G..rab -> 0
    # M..oveObj -> 1
    # P..lace -> 2
    # R..each -> 3
    # Generate numeric feature vector via LabelEncoder()
    feat_labenc = le.fit_transform(np.array(feapool))
    # Generate OneHot feature matrix via OneHotEncoder()
    feat_onehot = ohe.fit_transform(feat_labenc.reshape(len(feat_labenc), 1))
    print("Final dataset dimensions: " + str(data_raw.shape))
    return data_raw, feat_onehot


def gen_file_set(expr_set):
    fileset = []
    if len(expr_set) == 2:
        return EXPERIMENTS_DIR + "/AnnotationExperiment" + expr_set[1] + "MergedTable.txt"
    for num in expr_set:
        fileset.append(EXPERIMENTS_DIR + "/AnnotationExperiment" + num[1] + "MergedTable.txt")
    return fileset


def run_keras_cnn_model(loso_, epochs_n, run_suffix, aug_list):
    modelname = DATASET_NAME + ' CNN LOSO #' + loso_[4:]
    augmentations_ = aug_list
    batch_size_base = 15
    # regul_val = 0.015

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type = 'train.' + loso_
    histsave = OUTPUT_SAVES + DATASET_NAME + '_mnistcnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    weightsave = OUTPUT_SAVES + DATASET_NAME + '_mnistcnn_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + DATASET_NAME + '_mnistcnn_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

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

    training_generator = DataGenerator(DATASET_NAME, generator_type, batch_size_aug, ishape, list_idxes, augmentations_)

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
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    cnn_model.summary()
    print(datetime.now())
    print("Start training")
    history = cnn_model.fit_generator(generator=training_generator,
                                      epochs=epochs_n,
                                      shuffle=False, use_multiprocessing=False,  # CHANGE ON LINUX
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
    modelname = DATASET_NAME + ' LSTM LOSO #' + loso_[4:]
    augmentations_ = aug_list
    batch_size_base = 3
    # regul_val = 0.015

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = OUTPUT_SAVES + DATASET_NAME + '_lstm_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    weightsave = OUTPUT_SAVES + DATASET_NAME + '_lstm_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + DATASET_NAME + '_lstm_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data_.shape[0], 1)
    batch_size_aug = len(augmentations_) * batch_size_base
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    # print("Input Shape = %s " % (ishape,))

    training_generator = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug, ishape, list_idxes, augmentations_)

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    lstm_model = Sequential()
    lstm_model.add(Permute((2, 1, 3), input_shape=ishape))
    permute_shape = lstm_model.layers[0].output_shape
    resh_dim1 = permute_shape[2]
    resh_dim2 = permute_shape[1] * permute_shape[3]
    resh_shape = (resh_dim1, resh_dim2)
    lstm_model.add(Reshape(resh_shape))
    lstm_model.add(LSTM(128, input_shape=[batch_size_aug, resh_dim1, resh_dim2], stateful=False, unroll=True))
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
                                       shuffle=False, use_multiprocessing=True, 
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
    modelname = DATASET_NAME + ' ConvRNN LOSO #' + loso_[4:]
    augmentations_ = aug_list
    batch_size_base = 3
    regul_val = 0.015

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = OUTPUT_SAVES + DATASET_NAME + '_convrnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    weightsave = OUTPUT_SAVES + DATASET_NAME + '_convrnn_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + DATASET_NAME + '_convrnn_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data_.shape[0], 1)
    batch_size_aug = len(augmentations) * batch_size_base
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    # print("Input Shape = %s " % (ishape, ))

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    training_generator = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug, ishape, list_idxes, augmentations_)

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
    convrnn_model.add(LSTM(100, return_sequences=True, batch_input_shape=(100, 35, 200), kernel_regularizer=regularizers.l2(regul_val)))
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
    modelname = DATASET_NAME + ' ConvRNN LOSO #' + loso_[4:]
    cnn_batch_size_base = 20
    augmentations_ = aug_list
    # batch_size_base = 3
    regul_val = 0.015

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    rnn_histsave = OUTPUT_SAVES + DATASET_NAME + '_nunez_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    rnn_weightsave = OUTPUT_SAVES + DATASET_NAME + '_nunez_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + DATASET_NAME + '_nunez_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data_.shape[0], 1)
    batch_size_aug_cnn = len(augmentations) * cnn_batch_size_base
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    # print("Input Shape = %s " % (ishape, ))
    bi_shape = (batch_size_aug_cnn, test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    print("Batch Input Shape = %s " % (bi_shape,))

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    training_generator_cnn = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug_cnn, ishape, list_idxes, augmentations_)

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
    conv_model.layers.pop()  # Dense(Softmax)
    conv_model.layers.pop()  # Flatten()
    conv_model.layers.pop()  # Dense(100)
    conv_model.layers.pop()  # Dense(300)
    print("Continue with RNN")


    # RNN part
    rnn_batch_size_base = cnn_batch_size_base * 9
    batch_size_aug_rnn = len(augmentations) * rnn_batch_size_base

    training_generator_rnn = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug_rnn, ishape, list_idxes, augmentations_)

    # for layer in convrnn_model.layers:
    #     print(layer.output_shape)

    nunez_model = Sequential()
    nunez_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=ishape, #batch_input_shape=bi_shape,
                             padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    nunez_model.add(MaxPooling2D(pool_size=(2, 2)))
    nunez_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    nunez_model.add(MaxPooling2D(pool_size=(2, 2)))
    nunez_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    nunez_model.add(MaxPooling2D(pool_size=(2, 2)))

    nunez_model.set_weights(conv_model.get_weights())

    nunez_model.add(Permute((2, 1, 3)))
    #print(nunez_model.layers[-1].output_shape)
    nunez_model.add(Reshape((62, 200)))
    #print(nunez_model.layers[-1].output_shape)
    nunez_model.add(LSTM(100, return_sequences=True, stateful=False, unroll=True))  #batch_input_shape=(100, 25, 200), , kernel_regularizer=regularizers.l2(regul_val)
    nunez_model.add(Flatten())
    nunez_model.add(Dense(NUM_CLASSES, activation='softmax'))

    nunez_model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.993, decay=0.0),
                        metrics=['accuracy'])

    nunez_model.summary()
    print(datetime.now())
    print("Start training")
    history_rnn = nunez_model.fit_generator(generator=training_generator_rnn,
                                            epochs=int(epochs_n*5),
                                            shuffle=False, use_multiprocessing=True,
                                            validation_data=(test_data_, test_labels_),
                                            callbacks=[tensorboard])

    print(datetime.now())
    print(test_data_.shape)
    scores = nunez_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug_cnn)
    print(datetime.now())

    print("# KERAS MODEL: " + modelname + " # # # ")
    print('Test loss: %.4f' % scores[0])
    print('Test accuracy: %.3f %%' % (scores[1] * 100))
    pred_labels = nunez_model.predict(test_data_, batch_size=batch_size_aug_cnn)
    # print("Prediction matrix data:")
    # print(pred_labels.shape)
    # print(pred_labels)
    # print(datetime.now())

    with open(rnn_histsave, 'wb') as file_pi:
        pickle.dump(history_rnn.history, file_pi)
        print("Saved training history %s" % rnn_histsave)

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


CLMNS_JOINTS = [
    'boneCenter000X',
    'boneCenter000Y',
    'boneCenter000Z',
    'boneCenter001X',
    'boneCenter001Y',
    'boneCenter001Z',
    'boneCenter002X',
    'boneCenter002Y',
    'boneCenter002Z',
    'boneCenter003X',
    'boneCenter003Y',
    'boneCenter003Z',
    'boneCenter010X',
    'boneCenter010Y',
    'boneCenter010Z',
    'boneCenter011X',
    'boneCenter011Y',
    'boneCenter011Z',
    'boneCenter012X',
    'boneCenter012Y',
    'boneCenter012Z',
    'boneCenter013X',
    'boneCenter013Y',
    'boneCenter013Z',
    'boneCenter020X',
    'boneCenter020Y',
    'boneCenter020Z',
    'boneCenter021X',
    'boneCenter021Y',
    'boneCenter021Z',
    'boneCenter022X',
    'boneCenter022Y',
    'boneCenter022Z',
    'boneCenter023X',
    'boneCenter023Y',
    'boneCenter023Z',
    'boneCenter030X',
    'boneCenter030Y',
    'boneCenter030Z',
    'boneCenter031X',
    'boneCenter031Y',
    'boneCenter031Z',
    'boneCenter032X',
    'boneCenter032Y',
    'boneCenter032Z',
    'boneCenter033X',
    'boneCenter033Y',
    'boneCenter033Z',
    'boneCenter040X',
    'boneCenter040Y',
    'boneCenter040Z',
    'boneCenter041X',
    'boneCenter041Y',
    'boneCenter041Z',
    'boneCenter042X',
    'boneCenter042Y',
    'boneCenter042Z',
    'boneCenter043X',
    'boneCenter043Y',
    'boneCenter043Z',
    'wristPosition0X',
    'wristPosition0Y',
    'wristPosition0Z',
    'elbowPosition0X',
    'elbowPosition0Y',
    'elbowPosition0Z'
]

CLMNS_LABL_FIN = [
    'fineAnnotation'
]

VALID_LABELS = ["reach", "grab", "moveObject", "place"]

# Total of 50 files
EXPR_SET = [
    51,  1,  2,  4,  5,  6,  7,  9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50
]

batch_names = [
    ('kfold0', '51'),
    ('kfold0', '1'),
    ('kfold0', '2'),
    ('kfold0', '4'),
    ('kfold0', '5'),
    ('kfold0', '6'),
    ('kfold0', '7'),
    ('kfold0', '8'),
    ('kfold0', '9'),
    ('kfold0', '10'),
    ('kfold1', '11'),
    ('kfold1', '12'),
    ('kfold1', '13'),
    ('kfold1', '14'),
    ('kfold1', '15'),
    ('kfold1', '16'),
    ('kfold1', '17'),
    ('kfold1', '18'),
    ('kfold1', '19'),
    ('kfold1', '20'),
    ('kfold2', '21'),
    ('kfold2', '22'),
    ('kfold2', '23'),
    ('kfold2', '24'),
    ('kfold2', '25'),
    ('kfold2', '26'),
    ('kfold2', '27'),
    ('kfold2', '28'),
    ('kfold2', '29'),
    ('kfold2', '30'),
    ('kfold3', '31'),
    ('kfold3', '32'),
    ('kfold3', '33'),
    ('kfold3', '34'),
    ('kfold3', '35'),
    ('kfold3', '36'),
    ('kfold3', '37'),
    ('kfold3', '38'),
    ('kfold3', '39'),
    ('kfold3', '40'),
    ('kfold4', '41'),
    ('kfold4', '42'),
    ('kfold4', '43'),
    ('kfold4', '44'),
    ('kfold4', '45'),
    ('kfold4', '46'),
    ('kfold4', '47'),
    ('kfold4', '48'),
    ('kfold4', '49'),
    ('kfold4', '50')
]

DATASET_NAME = 'AVC'
# ACTIONS: GRAB, MOVE, PLACE, REACH
NUM_CLASSES = 4

# PARAMETERS #
iterations = 1
num_epochs = 1
# AUGMENTATIONS: none, shift, scale, noise, subsample, interpol
augmentations = [
    'none',
    # 'shift',
    # 'scale',
    'noise',
    # 'subsample',
    # 'interpol'
]
EXPERIMENTS_DIR = "./experimentsData/"
# SET OUTPUT_SAVES OUTSIDE THE DOCKER CONTAINER
OUTPUT_SAVES = "."
# MODELS: CNN, LSTM, ConvRNN
train_models = [
   'CNN',
#    'LSTM',
#     'ConvRNN'
]
# END OF PARAMETERS

le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

sample_filelist = gen_file_set(batch_names)

for model in train_models:
    for run in np.arange(0, iterations, 1):
        for key, batch_group in itertools.groupby(batch_names, lambda x: x[0]):
        # for sample, sam_num in zip(sample_filelist, EXPR_SET):
        #     loso = "loso" + str(sam_num)
            print("Batch: " + key)
            print("Augmentations: %s" % augmentations)

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
                #run_keras_convrnn_model(loso, num_epochs, str(int(run + 1)), augmentations)
                run_keras_nunez_model(key, num_epochs, str(int(run + 1)), augmentations)
            else:
                print("Model unknown!")

        print("Finished run #" + str(int(run + 1)))

exit(0)
