import keras
import random
import numpy as np
# import sklearn
from numpy import matlib
from sklearn.utils import shuffle


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, name_, type_, batch_size, dim, list_idxes, aug_types):
        """Initialization"""
        self.name = name_
        self.type = type_
        self.batch_size = int(batch_size / len(aug_types))
        self.dim = dim
        self.list_idxes = list_idxes
        self.aug_types = aug_types
        self.train_data = np.load(self.name + '.' + self.type + '.data.npy')
        self.train_labels = np.load(self.name + '.' + self.type + '.labels.npy')
        self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels)
        self.sigma = (0.1 / 4)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.list_idxes[index*self.batch_size:(index+1)*self.batch_size]

        #print("Processing batch #%d" % index)
        #print(indexes)
        # Generate data
        x, y = self.__data_generation(indexes)

        #print(" %s getitem() returns x in %s and y in %s " % (self.type, (x.shape, ), (y.shape, )))
        #print("y dist: {}".format(np.sum(y, axis=0)))
        return x, y

    def on_epoch_end(self):
        #"""Updates indexes after each epoch"""
        #print("Epochs end.")
        self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        #print("LEN method called %s " % self.type)
        return int(np.floor(len(self.list_idxes) / (self.batch_size * len(self.aug_types))))

    def scale_augmentation(self, odata):
        #print("Doing data scale augmentation.")
        # fascale = random.gauss(mu=1, sigma=self.sigma)  # Nunez Aug
        fascale = 1 + (random.randrange(-20, 20, 1) / 100)  # uniform distribution
        #print("Scale factor: %.2f" % fascale)
        return np.multiply(odata, fascale)

    def shift_augmentation(self, odata):
        max_width = self.dim[1]
        #print("Doing data shift augmentation.")
        shift_x_fac = random.gauss(mu=0, sigma=self.sigma)  # Nunez
        shift_y_fac = random.gauss(mu=0, sigma=self.sigma)  # Nunez
        #print("Shift factors (dx,dy) = (%.3f, %.3f)" % (shift_x_fac, shift_y_fac))
        # Do the shift augmentation
        shift_vec = np.ones([1, 3])
        shift_vec[0, 0] = shift_vec[0, 0] * shift_x_fac
        shift_vec[0, 1] = shift_vec[0, 1] * shift_y_fac
        shift_vec[0, 2] = shift_vec[0, 2] * 0  # empty clmn
        #print(shift_vec.shape)
        #print(shift_vec)
        shift = matlib.repmat(shift_vec, max_width, 1)
        #print(shift.shape)
        return  odata + shift

    def noise_augmentation(self, odata):
        max_width = self.dim[1]
        #print("Doing data noise augmentation.")
        data_aug = np.zeros([odata.shape[0], odata.shape[1], odata.shape[2], odata.shape[3]])
        # Data augmentation: by noise
        num_seqncs = odata.shape[0]
        # print("num_frames %s " % num_frames)
        num_joints = odata.shape[1]
        joints_total = random.randint(1, 4)  # random num of joints to augment
        joints_range = list(np.arange(0, num_joints, 1))
        noise_joints = []
        for x in np.arange(0, joints_total, 1):
            id_j = random.choice(joints_range)
            noise_joints.append(id_j)
            joints_range.remove(id_j)
        # print(noise_joints)
        # for f in range(0, num_frames, 1):
        for seq in range(num_seqncs):
            # for j in range(0, num_joints, 1):
            for jnt in range(num_joints):
                #print(" %d %d " % (f,n))
                if jnt in noise_joints:
                    noise_factors = np.ones([max_width, 3])
                    noise_factors[:, 0] = noise_factors[:, 0] * random.gauss(mu=0, sigma=self.sigma)  # Nunez
                    noise_factors[:, 1] = noise_factors[:, 1] * random.gauss(mu=0, sigma=self.sigma)  # Nunez
                    noise_factors[:, 2] = noise_factors[:, 2] * random.gauss(mu=0, sigma=self.sigma)  # Nunez
                    data_aug[seq, jnt, :, :] = odata[seq, jnt, :, :] + noise_factors
                    # print("Noise factors: ")
                    # print(noise_factors)
                else:
                    data_aug[seq, jnt, :, :] = odata[seq, jnt, :, :]

    def subsample_augmentation(self, odata):
        #print("Doing data subsample augmentation.")
        data_aug = np.zeros([odata.shape[0], odata.shape[1], odata.shape[2], odata.shape[3]])
        num_seqncs = odata.shape[0]
        num_frames = odata.shape[2]
        # Alternative go with all possible combination would be for d in (2,3,4) and for m in (2,3)
        # better generate more random epochs? #17.10.2018
        for seq in range(0, num_seqncs, 1):
            d = random.randint(2, 4)  # random displacement to sequal (2, 3, 4)
            m = random.randint(2, 3)  # random step to iterate (2, 3)
            #print("Subsample %d random numbers d = %d, m = %d" % (f,d,m))
            for frm, p in zip(range(d, num_frames, m), range(0, num_frames, 1)):
                #print(s, p)
                data_aug[seq, :, p, :] = odata[seq, :, frm, :]
        return data_aug

    def interpolate_augmentation(self, odata):
        # print("Doing time interpolation data augmentation")
        data_aug = np.zeros([odata.shape[0], odata.shape[1], odata.shape[2], odata.shape[3]])
        num_seqncs = odata.shape[0]
        num_joints = odata.shape[1]
        num_frames = odata.shape[2]
        for seq in range(num_seqncs):
            r = random.randint(20, 80) / 100
            # print("Random scaling factor: %f" % r)
            for jnt in range(num_joints):
                for frm in range(num_frames):
                    # print(" f=%d j=%d s=%d " % (seq, jnt, frm))
                    # print("Current coordinate values: %s " % odata[seq, jnt, frm, :])
                    # print("Next  coordinate values: %s " % odata[seq, jnt, int(frm + 1), :])
                    frm_prev = odata[seq, jnt, frm, :]
                    frm_next = odata[seq, jnt, int(frm + 1), :]
                    if (frm_prev == 0.0).all() and (frm_next == 0.0).all():
                        # print("Interpolation break - padding reached")
                        break
                    frm_step = np.subtract(frm_next, frm_prev)

                    data_aug[seq, jnt, int(frm + 1), :] = np.add(frm_prev, np.multiply(frm_step, r))
                    # print("Interpolated coordinate value: %s " % data_aug[seq, jnt, int(frm + 1), :])
        return data_aug

    def __augment_data(self, augtype, odata):
        if augtype == 'none':
            return odata
        elif augtype == 'scale':
            return self.scale_augmentation(odata)
        elif augtype == 'shift':
            return self.shift_augmentation(odata)
        elif augtype == 'noise':
            return self.noise_augmentation(odata)
        elif augtype == 'subsample':
            return self.subsample_augmentation(odata)
        elif augtype == 'interpol':
            return self.interpolate_augmentation(odata)
        elif augtype == 'scale_shift':
            scaled = self.scale_augmentation(odata)
            return self.shift_augmentation(scaled)

    def __data_generation(self, idxes):
        # print("Calling data_generation()")
        augmentations = self.aug_types

        train_data = self.train_data[idxes]
        # print(train_data.shape)
        train_labels = self.train_labels[idxes]

        # Generate augmentation data
        agmnt_data = []
        agmnt_labels = []

        for agmt_type in augmentations:
            sppl_data = self.__augment_data(agmt_type, train_data)
            aug_data, aug_labels = shuffle(sppl_data, train_labels, random_state=42)
            agmnt_data.append(aug_data)
            agmnt_labels.append(aug_labels)

        batch_data = np.asarray(agmnt_data).reshape(-1, train_data.shape[1], train_data.shape[2], train_data.shape[3])
        batch_labels = np.asarray(agmnt_labels).reshape(-1, train_labels.shape[1])
        # Shuffle the whole batch before returning
        batch_data_fin, batch_labels_fin = shuffle(batch_data, batch_labels)
        # print("dataGen() returns batch_data of %s and batch_labels of %s " % ((batch_data_fin.shape, ), (batch_labels_fin.shape, )))
        return batch_data_fin, batch_labels_fin
