from __future__ import print_function
import keras
import pickle
import math
import os
import time
import numpy as np
import itertools
from datetime import datetime
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Permute
from keras.layers import Conv2D, MaxPooling2D, LSTM, Masking
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from DataGenerator import DataGenerator

# NOTE: Dataset from http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html is of bad quality.
# FrameIDs are missing, doubled or false described in actionLabel
# Values are corrected to authors best knowledge


# def normalize_data(data_):
#     orig_shape = data_.shape
#     data_ = np.reshape(data_, [orig_shape[0], -1])
#     data_ = StandardScaler().fit_transform(data_)
#     # data_ = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data_)
#     data_ = np.reshape(data_, orig_shape)
#     return data_


def skeleton_reshape(frame_):
    num_joints = 20
    avg_length = 30
    ext_step = 4
    ext_factors = [0, 0.25, 0.5, 0.75]
    new_frame = np.zeros([num_joints, MAX_WIDTH, 3])
    frame_length = frame_.shape[0]
    if frame_length < avg_length:
        for frame_id in range(int(frame_length-1)):
            for joint in range(num_joints):
                frm_current = frame_[frame_id, joint:joint + 3]
                frm_next = frame_[int(frame_id + 1), joint: joint + 3]
                frm_step = np.subtract(frm_next, frm_current)
                for eid in range(ext_step):
                    new_frame[joint, int(frame_id * ext_step + eid), :] = \
                        np.add(frm_current, np.multiply(frm_step, ext_factors[eid]))
    else:
        for frame_id in range(frame_length):
            for joint in range(num_joints):
                new_frame[joint, frame_id, :] = frame_[frame_id, joint: joint + 3]
    return new_frame


def process_sample(sample_name):
    data_set = []
    labels = []

    sample_ = np.loadtxt(sample_name)
    sample_data = sample_[:, 1:]
    sample_ids = np.array(sample_[:, 0]).astype(int).reshape([-1, 1])

    # Load ID range for action Walk [0]
    walk_params = content[l+1].split(" ")
    walk_min = int(np.where(sample_ids == int(walk_params[1].lstrip()))[0])
    walk_max = int(np.where(sample_ids == int(walk_params[2].lstrip()))[0]+1)
    walk_sample = skeleton_reshape(sample_data[walk_min: walk_max, :])
    data_set.append(walk_sample)
    labels.append(0)
    # print("Walk Sequence Length = %d Frames" % (walk_max - walk_min))

    # Load ID range for action SitDown [1]
    sitdown_params = content[l+2].split(" ")
    sitdown_min = int(np.where(sample_ids == int(sitdown_params[1].lstrip()))[0])
    sitdown_max = int(np.where(sample_ids == int(sitdown_params[2].lstrip()))[0]+1)
    sitdown_sample = skeleton_reshape(sample_data[sitdown_min: sitdown_max, :])
    data_set.append(sitdown_sample)
    labels.append(1)
    # print("SitDown Sequence Length = %d Frames" % (sitdown_max - sitdown_min))

    # Load ID range for action StandUp [2]
    standup_params = content[l+3].split(" ")
    standup_min = int(np.where(sample_ids == int(standup_params[1].lstrip()))[0])
    standup_max = int(np.where(sample_ids == int(standup_params[2].lstrip()))[0]+1)
    standup_sample = skeleton_reshape(sample_data[standup_min: standup_max, :])
    data_set.append(standup_sample)
    labels.append(2)
    # print("StandUp Sequence Length = %d Frames" % (standup_max - standup_min))

    # Load ID range for action PickUp [3]
    pickup_params = content[l+4].split(" ")
    pickup_min = int(np.where(sample_ids == int(pickup_params[1].lstrip()))[0])
    pickup_max = int(np.where(sample_ids == int(pickup_params[2].lstrip()))[0]+1)
    pickup_sample = skeleton_reshape(sample_data[pickup_min: pickup_max, :])
    data_set.append(pickup_sample)
    labels.append(3)
    # print("PickUp Sequence Length = %d Frames" % (pickup_max - pickup_min))

    # Load ID range for action Carry [4]
    carry_params = content[l+5].split(" ")
    if 'NaN' in carry_params:
        print("Carry Value NaN detected. Filling in blanks.")
        data_set.append(np.zeros([20, MAX_WIDTH, 3]))
        labels.append(4)
    else:
        carry_min = int(np.where(sample_ids == int(carry_params[1].lstrip()))[0])
        carry_max = int(np.where(sample_ids == int(carry_params[2].lstrip()))[0] + 1)
        carry_sample = skeleton_reshape(sample_data[carry_min: carry_max, :])
        data_set.append(carry_sample)
        labels.append(4)
        # print("Carry Sequence Length = %d Frames" % (carry_max - carry_min))

    # Load ID range for action Throw [5]
    throw_params = content[l+6].split(" ")
    throw_min = int(np.where(sample_ids == int(throw_params[1].lstrip()))[0])
    throw_max = int(np.where(sample_ids == int(throw_params[2].lstrip()))[0]+1)
    throw_sample = skeleton_reshape(sample_data[throw_min: throw_max, :])
    data_set.append(throw_sample)
    labels.append(5)
    # print("Throw Sequence Length = %d Frames" % (throw_max - throw_min))

    # Load ID range for action Push [6]
    push_params = content[l+7].split(" ")
    push_min = int(np.where(sample_ids == int(push_params[1].lstrip()))[0])
    push_max = int(np.where(sample_ids == int(push_params[2].lstrip()))[0]+1)
    push_sample = skeleton_reshape(sample_data[push_min: push_max, :])
    data_set.append(push_sample)
    labels.append(6)
    # print("Push Sequence Length = %d Frames" % (push_max - push_min))

    # Load ID range for action Pull [7]
    pull_params = content[l+8].split(" ")
    pull_min = int(np.where(sample_ids == int(pull_params[1].lstrip()))[0])
    pull_max = int(np.where(sample_ids == int(pull_params[2].lstrip()))[0]+1)
    pull_sample = skeleton_reshape(sample_data[pull_min: pull_max, :])
    data_set.append(pull_sample)
    labels.append(7)
    # print("Pull Sequence Length = %d Frames" % (pull_max-pull_min))

    # Load ID range for action WaveHands [8]
    wavehands_params = content[l+9].split(" ")
    wavehands_min = int(np.where(sample_ids == int(wavehands_params[1].lstrip()))[0])
    wavehands_max = int(np.where(sample_ids == int(wavehands_params[2].lstrip()))[0]+1)
    wavehands_sample = skeleton_reshape(sample_data[wavehands_min: wavehands_max, :])
    data_set.append(wavehands_sample)
    labels.append(8)
    # print("WaveHands Sequence Length = %d Frames" % (wavehands_max - wavehands_min))

    # Load ID range for action ClapHands [9]
    claphands_params = content[l+10].split(" ")
    claphands_min = int(np.where(sample_ids == int(claphands_params[1].lstrip()))[0])
    claphands_max = int(np.where(sample_ids == int(claphands_params[2].lstrip()))[0]+1)
    claphands_sample = skeleton_reshape(sample_data[claphands_min: claphands_max, :])
    data_set.append(claphands_sample)
    labels.append(9)
    # print("ClapHands Sequence Length = %d Frames" % (claphands_max - claphands_min))

    # print("end of sample routine")
    return np.asarray(data_set), np.asarray(labels).reshape([-1, 1])


def run_keras_cnn_model(loso_, epochs_n, run_suffix, aug_list):
    modelname = DATASET_NAME + ' CNN LOSO #' + loso_[4:]
    batch_size_base = 5
    # regul_val = 0.015
    augmentations_ = aug_list

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type = 'train.' + loso_
    histsave = OUTPUT_SAVES + DATASET_NAME + '_mnistcnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    scoresave = OUTPUT_SAVES + DATASET_NAME + '_mnistcnn_scores.' + loso_ + '.' + run_suffix + '.save'
    weightsave = OUTPUT_SAVES + DATASET_NAME + '_mnistcnn_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + DATASET_NAME + '_mnistcnn_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data = np.load(test_data_file_)
    test_labels = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data.shape[0], 1)
    batch_size_aug = len(augmentations_) * batch_size_base
    ishape = (test_data.shape[1], test_data.shape[2], test_data.shape[3])
    # print("Input Shape = %s " % (ishape, ))

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    # Generators
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
                                      shuffle=False, use_multiprocessing=False,  # CHANGE ON RACER!
                                      callbacks=[tensorboard])

    # history = model.fit(train_data, train_labels,
    #                     batch_size=batch_size_aug,
    #                     epochs=epochs_n, verbose=1,
    #                     validation_data=(test_data, test_labels))

    print(datetime.now())
    print(test_data.shape)
    scores = cnn_model.evaluate(test_data, test_labels, batch_size=batch_size_aug)
    print(datetime.now())

    print("# KERAS MODEL: " + modelname + " # # # ")
    print('Test loss: %.4f' % scores[0])
    print('Test accuracy: %.3f %%' % (scores[1] * 100))
    pred_labels = cnn_model.predict(test_data, batch_size=batch_size_aug)
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

    cnf_matrix = confusion_matrix(test_labels.argmax(axis=1), pred_labels.argmax(axis=1))
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
    batch_size_base = 5
    regul_val = 0.015
    augmentations_ = aug_list

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = OUTPUT_SAVES + DATASET_NAME + '_lstm_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    scoresave = OUTPUT_SAVES + DATASET_NAME + '_lstm_scores.' + loso_ + '.' + run_suffix + '.save'
    weightsave = OUTPUT_SAVES + DATASET_NAME + '_lstm_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + DATASET_NAME + '_lstm_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data.shape[0], 1)
    batch_size_aug = len(augmentations_) * batch_size_base
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    timesteps_dim = test_data_.shape[2]
    features_dim = test_data_.shape[1] * test_data_.shape[3]
    # print("Input Shape = %s " % (ishape,))

    # Generators
    training_generator = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug, ishape, list_idxes, augmentations_)

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    lstm_model = Sequential()
    lstm_model.add(Permute((2, 1, 3), input_shape=ishape,  batch_input_shape=(batch_size_aug, test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])))
    permute_shape = lstm_model.layers[0].output_shape
    resh_dim1 = permute_shape[2]
    resh_dim2 = permute_shape[1] * permute_shape[3]
    resh_shape = (resh_dim1, resh_dim2)
    lstm_model.add(Reshape(resh_shape))
    lstm_model.add(LSTM(128, return_sequences=True, stateful=True, unroll=True, kernel_regularizer=regularizers.l2(regul_val))) #input_shape=[batch_size_aug, resh_dim1, resh_dim2],
    lstm_model.add(LSTM(128, return_sequences=True, stateful=True))
    lstm_model.add(LSTM(128, stateful=True))
    # lstm_model.add(Dense(128, activation='relu'))
    # lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(NUM_CLASSES, activation='softmax'))

    # lstm_model = Sequential()
    # lstm_model.add(LSTM(128, return_sequences=True, stateful=False, batch_input_shape=(batch_size_aug, timesteps_dim, features_dim)))
    # lstm_model.add(LSTM(128, return_sequences=True, stateful=False))
    # lstm_model.add(LSTM(128, stateful=False))
    # lstm_model.add(Dense(NUM_CLASSES, activation='softmax'))

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
    # # print(test_data_.shape)
    # test_data_, test_labels_ = trim_to_batch(test_data_, test_labels_, batch_size_aug)
    # # print(test_data_.shape)
    # test_data_ = test_data_.reshape((-1, timesteps_dim, features_dim))

    scores = lstm_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug)
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
    modelname = DATASET_NAME + ' ConvRNN LOSO #' + loso_[4:]
    batch_size_base = 10  #special
    regul_val = 0.015
    augmentations_ = aug_list

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    histsave = OUTPUT_SAVES + DATASET_NAME + '_convrnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    scoresave = OUTPUT_SAVES + DATASET_NAME + '_convrnn_scores.' + loso_ + '.' + run_suffix + '.save'
    weightsave = OUTPUT_SAVES + DATASET_NAME + '_convrnn_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + DATASET_NAME + '_convrnn_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

    print("Loading data from saved files.")
    train_data_ = np.load(train_data_file_)
    # train_labels_ = np.load(train_labels_file_)
    test_data_ = np.load(test_data_file_)
    print("Test Data Shape = %s " % (test_data_.shape,))
    test_labels_ = np.load(test_labels_file_)

    list_idxes = np.arange(0, len(augmentations_) * train_data_.shape[0], 1)
    batch_size_aug = len(augmentations) * batch_size_base
    ishape = (test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    # print("Input Shape = %s " % (ishape, ))
    bi_shape = (batch_size_aug, test_data_.shape[1], test_data_.shape[2], test_data_.shape[3])
    print("Batch Input Shape = %s " % (bi_shape,))

    tensorboard = TensorBoard(log_dir='.', histogram_freq=0,
                              write_graph=True, write_images=False)

    # Generators
    training_generator = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug, ishape, list_idxes, augmentations_)

    convrnn_model = Sequential()
    convrnn_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=ishape, #batch_input_shape=bi_shape,
                             padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    convrnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    convrnn_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    convrnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    convrnn_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    convrnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    # CNN part
    # convrnn_model.add(Dense(300))
    # convrnn_model.add(Dense(100))

    # for layer in convrnn_model.layers:
    #     print(layer.output_shape)

    # RNN part
    convrnn_model.add(Permute((2, 1, 3)))
    print(convrnn_model.layers[-1].output_shape)
    convrnn_model.add(Reshape((25, 200)))
    print(convrnn_model.layers[-1].output_shape)
    convrnn_model.add(LSTM(100, return_sequences=True, stateful=False))  #batch_input_shape=(100, 25, 200), , kernel_regularizer=regularizers.l2(regul_val)
    # model.add(Dropout(0.5))
    convrnn_model.add(Flatten())  # needed only for return_sequences=True
    convrnn_model.add(Dense(NUM_CLASSES, activation='softmax'))

    convrnn_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

    convrnn_model.summary()
    print(datetime.now())
    print("Start training")
    history = convrnn_model.fit_generator(generator=training_generator,
                                          epochs=epochs_n,
                                          shuffle=False, use_multiprocessing=False,  # CHANGE ON RACER!
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

    with open(scoresave, 'wb') as file_pi:
        pickle.dump(scores, file_pi)
        print("Saved training history %s" % scoresave)

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
    modelname = DATASET_NAME + ' Nunez LOSO #' + loso_[4:]
    cnn_batch_size_base = 20
    # epochs_n_cnn = 100
    regul_val = 0.015
    augmentations_ = aug_list

    train_data_file_ = DATASET_NAME + ".train." + loso_ + ".data.npy"
    # train_labels_file_ = DATASET_NAME + ".train." + loso_ + ".labels.npy"
    test_data_file_ = DATASET_NAME + ".test." + loso_ + ".data.npy"
    test_labels_file_ = DATASET_NAME + ".test." + loso_ + ".labels.npy"
    generator_type_train = 'train.' + loso_
    cnn_histsave = OUTPUT_SAVES + DATASET_NAME + '_nunez_cnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    rnn_histsave = OUTPUT_SAVES + DATASET_NAME + '_nunez_rnn_trainHistoryDict.' + loso_ + '.' + run_suffix + '.save'
    cnn_scoresave = OUTPUT_SAVES + DATASET_NAME + '_nunez_cnn_scores.' + loso_ + '.' + run_suffix + '.save'
    rnn_scoresave = OUTPUT_SAVES + DATASET_NAME + '_nunez_rnn_scores.' + loso_ + '.' + run_suffix + '.save'
    rnn_weightsave = OUTPUT_SAVES + DATASET_NAME + '_nunez_weights_' + loso_ + '.' + run_suffix + '.h5'
    cfsave = OUTPUT_SAVES + DATASET_NAME + '_nunez_confusion_matrix_' + loso_ + '.' + run_suffix + '.save'

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
    training_generator_cnn = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug_cnn, ishape, list_idxes, augmentations_)

    conv_model = Sequential()
    # conv_model.add(Masking(mask_value=0.0))
    conv_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=ishape, #batch_input_shape=bi_shape,
                             padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    conv_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    conv_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    # CNN part
    conv_model.add(Dense(300))
    conv_model.add(Dense(100))
    conv_model.add(Flatten())
    conv_model.add(Dense(NUM_CLASSES, activation='softmax'))

    conv_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adadelta(),
                       metrics=['accuracy'])

    conv_model.summary()
    print(datetime.now())
    print("Start training")
    history_cnn = conv_model.fit_generator(generator=training_generator_cnn,
                                           epochs=epochs_n,
                                           shuffle=False, use_multiprocessing=False,  # CHANGE ON RACER!
                                           callbacks=[tensorboard])

    print(datetime.now())
    cnn_scores = conv_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug_cnn)
    print(datetime.now())

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


    # history = model.fit(train_data, train_labels,
    #                     batch_size=batch_size_aug,
    #                     epochs=epochs_n, verbose=1,
    #                     validation_data=(test_data, test_labels))

    # RNN part
    rnn_batch_size_base = 6
    batch_size_aug_rnn = len(augmentations) * rnn_batch_size_base

    training_generator_rnn = DataGenerator(DATASET_NAME, generator_type_train, batch_size_aug_rnn, ishape, list_idxes, augmentations_)
    # convrnn_model.build()
    # convrnn_model.summary()
    # convrnn_model.compile(loss=keras.losses.categorical_crossentropy,
    #                       optimizer=keras.optimizers.Adadelta(),
    #                       metrics=['accuracy'])
    # for layer in convrnn_model.layers:
    #     print(layer.output_shape)

    nunez_model = Sequential()

    nunez_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=ishape, #batch_input_shape=bi_shape,
                             padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    nunez_model.add(MaxPooling2D(pool_size=(2, 2)))
    # nunez_model.add(Dropout(0.5))
    nunez_model.add(Conv2D(50, kernel_size=(2, 2), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    nunez_model.add(MaxPooling2D(pool_size=(2, 2)))
    # nunez_model.add(Dropout(0.5))
    nunez_model.add(Conv2D(100, kernel_size=(3, 3), activation='relu', padding='same'))  #, kernel_regularizer=regularizers.l2(regul_val)
    nunez_model.add(MaxPooling2D(pool_size=(2, 2)))

    nunez_model.set_weights(conv_model.get_weights())

    print(nunez_model.layers[-1].output_shape)
    nunez_model.add(Reshape((15, 200))) # 25 for framesize of 200 and 15 for 120
    print(nunez_model.layers[-1].output_shape)
    # nunez_model.add(Masking(mask_value=0.0))
    nunez_model.add(LSTM(100, return_sequences=True, stateful=False))  #batch_input_shape=(100, 25, 200), , kernel_regularizer=regularizers.l2(regul_val)
    # model.add(Dropout(0.5))
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
                                            shuffle=False, use_multiprocessing=False,  # CHANGE ON RACER!
                                            callbacks=[tensorboard])

    print(datetime.now())
    rnn_scores = nunez_model.evaluate(test_data_, test_labels_, batch_size=batch_size_aug_cnn)
    print(datetime.now())

    print("# KERAS MODEL: " + modelname + " # # # ")
    print('Test loss: %.4f' % rnn_scores[0])
    print('Test accuracy: %.3f %%' % (rnn_scores[1] * 100))
    pred_labels = nunez_model.predict(test_data_, batch_size=batch_size_aug_cnn)
    # print("Prediction matrix data:")
    # print(pred_labels.shape)
    # print(pred_labels)
    # print(datetime.now())

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
    ('kfold1', 's02_e01'),
    ('kfold1', 's02_e02'),
    ('kfold2', 's03_e01'),
    ('kfold2', 's03_e02'),
    ('kfold3', 's04_e01'),
    ('kfold3', 's04_e02'),
    ('kfold4', 's05_e01'),
    ('kfold4', 's05_e02'),
    ('kfold5', 's06_e01'),
    ('kfold5', 's06_e02'),
    ('kfold6', 's07_e01'),
    ('kfold6', 's07_e02'),
    ('kfold7', 's08_e01'),
    ('kfold7', 's08_e02'),
    ('kfold8', 's09_e01'),
    ('kfold8', 's09_e02'),
    ('kfold9', 's10_e01'),
    ('kfold9', 's10_e02')
]

DATASET_NAME = 'UTK'
# STATIC PARAMETERS
NUM_FILES = 20
LINE_STEP = 11
NUM_CLASSES = 10
MAX_WIDTH = 120
# EDITABLE PARAMETERS
DIRECTORY = "/home/antonk/racer/UTKinect3D/joints/"
UTKLABELSFILE = "/home/antonk/racer/UTKinect3D/actionLabel.txt"
# DIRECTORY = "D:\\!DA-20092018\\UTKinectAction3D\\joints\\"
# UTKLABELSFILE = "D:\\!DA-20092018\\UTKinectAction3D\\actionLabel.txt"
# SET OUTPUT_SAVES OUTSIDE THE DOCKER CONTAINER
OUTPUT_SAVES = "./"

iterations = 1
num_epochs = 1  # fix the .fit method as well
# AUGMENTATIONS: none, shift, scale, noise, subsample, interpol
augmentations = [
    'none',
    # 'shift',
    # 'scale',
    # 'noise',
    # 'subsample',
    'interpol'
]
# MODELS: CNN, LSTM, ConvRNN
train_models = [
    'CNN',
    # 'LSTM',
    # 'ConvRNN'
]
# END OF PARAMETERS

le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

actionLabels = open(UTKLABELSFILE, "r")
content = actionLabels.readlines()
actionLabels.close()

for model in train_models:
    for run in np.arange(iterations):
        for key, batch_group in itertools.groupby(batch_names, lambda x: x[0]):
            print("Batch: " + key)
            print("Augmentations: %s" % augmentations)

            train_data_file = DATASET_NAME + ".train." + key + ".data"
            train_labels_file = DATASET_NAME + ".train." + key + ".labels"
            test_data_file = DATASET_NAME + ".test." + key + ".data"
            test_labels_file = DATASET_NAME + ".test." + key + ".labels"

            if os.path.exists(train_data_file + ".npy") and os.path.exists(train_labels_file + ".npy") \
                    and os.path.exists(test_data_file + ".npy") and os.path.exists(test_labels_file + ".npy"):
                print("Sample TRAIN and TEST data sets files found. Skipping generation from skeleton data.")
            else:
                data_train = np.zeros([1, 20, MAX_WIDTH, 3])
                labels_train = np.zeros([1, 1])
                data_test = np.zeros([1, 20, MAX_WIDTH, 3])
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

                for l in range(0, 220, 11):
                    # print("Parsing start line: %d " % l)
                    filename = DIRECTORY + "joints_" + content[l].strip('\n').lstrip() + ".txt"
                    print("Parsing file: %s " % filename)
                    file_data, file_labels = process_sample(filename)

                    if any(test in filename for test in train_files):
                        data_train = np.append(data_train, file_data, axis=0)
                        labels_train = np.append(labels_train, file_labels, axis=0)
                    else:
                        data_test = np.append(data_test, file_data, axis=0)
                        labels_test = np.append(labels_test, file_labels, axis=0)

                data_train = data_train[1:, :, :, :]
                labels_train = labels_train[1:, :]
                data_test = data_test[1:, :, :, :]
                labels_test = labels_test[1:, :]

                # print(le.fit(np.asarray(labels_train)).classes_)
                utk_dataset_train = np.asarray(data_train)
                # utk_dataset_train = normalize_data(utk_dataset_train)
                feat_labenc = le.fit_transform(np.asarray(labels_train))
                utk_labels_train = ohe.fit_transform(feat_labenc.reshape(len(feat_labenc), 1))

                utk_dataset_test = np.asarray(data_test)
                # utk_dataset_test = normalize_data(utk_dataset_test)
                feat_labenc = le.fit_transform(np.asarray(labels_test))
                utk_labels_test = ohe.fit_transform(feat_labenc.reshape(len(feat_labenc), 1))

                utk_dataset_train, utk_labels_train = shuffle(utk_dataset_train, utk_labels_train, random_state=42)
                utk_dataset_test, utk_labels_test = shuffle(utk_dataset_test, utk_labels_test, random_state=42)

                print("Saving data & labels to files: ")
                print("TRAIN data  : %s" % train_data_file)
                print("TRAIN labels: %s" % train_labels_file)
                np.save(train_data_file, utk_dataset_train)
                np.save(train_labels_file, utk_labels_train)
                print("TEST data  : %s" % test_data_file)
                print("TEST labels: %s" % test_labels_file)
                np.save(test_data_file, utk_dataset_test)
                np.save(test_labels_file, utk_labels_test)

            print("Training " + model + " model")
            if model == 'CNN':
                run_keras_cnn_model(key, num_epochs, str(int(run + 1)), augmentations)
            elif model == 'LSTM':
                run_keras_lstm_model(key, num_epochs, str(int(run + 1)), augmentations)
            elif model == 'ConvRNN':
                run_keras_nunez_model(key, num_epochs, str(int(run + 1)), augmentations)
            else:
                print("Model unknown!")

        print("Finished run #" + str(int(run+1)))

exit(0)
