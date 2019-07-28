import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, Dense, Flatten, Reshape, Conv1D, Conv2D, Add, BatchNormalization, MaxPooling1D)
import numpy as np
from random import sample, shuffle

def resblock(input_, filters_, shortcut=False):
    """Creates a Residual Layer with x amount of filters. Returns the bottom
    layer.
    """
    conv1d_r1 = Conv1D(
                    filters=filters_,
                    kernel_size=3,
                    padding="same",
                    activation='relu'
                )(input_)
    conv1d_r2 = Conv1D(
                    filters=filters_,
                    kernel_size=3,
                    padding="same",
                    activation='relu'
                )(conv1d_r1)
    conv1d_r3 = Conv1D(
                    filters=filters_,
                    kernel_size=1,
                    padding="same",
                    activation='relu'
                )(conv1d_r2)
    maxpool_r = MaxPooling1D(pool_size=2)(conv1d_r3)

    if shortcut:
        conv1d_l = Conv1D(filters=filters_, kernel_size=3, padding="same")(input_)
        maxpool_l = MaxPooling1D(pool_size=2)(conv1d_l)

        add = Add()([maxpool_l, maxpool_r])
        return add

    return maxpool_r


def onehot(seq):
    """Return the amino acid sequence as one hot coded numpy array"""
    oh = np.zeros([longest, 20])

    for i, j in enumerate(seq):
        oh[i][aa2int[j]] = 1

    return oh


aa1 = list("ACDEFGHIKLMNPQRSTVWY")
aa2int = dict((c, i) for i, c in enumerate(aa1))

#Parsing of filtered dataset file
raw = list(open("../data/nodups.fasta", 'r'))
raw = [i[:-1] for i in raw] # remove newlines
labels = np.array([int(i.split(";")[1]) for i in raw])
seq = [i.split(";")[0] for i in raw]
data = list(zip(seq, labels))
longest = max([len(i[0]) for i in data]) # Currently 8797

#Equal number of allergens and non-allergens. Logs the data used.
allergens = [i for i in data if i[1] == 1]
nonallergens = [i for i in data if i[1] == 0] 
nonaller_sample = sample(nonallergens, len(allergens))

#Random selection of the data by shuffling the list
shuffle(allergens)
shuffle(nonaller_sample)
aller_train = allergens[:int(len(allergens)*0.8)]
aller_test = allergens[int(len(allergens)*0.8):int(len(allergens)*0.9)]
aller_val = allergens[int(len(allergens)*0.9):]
nonaller_train = nonaller_sample[:int(len(nonaller_sample)*0.8)]
nonaller_test = nonaller_sample[int(len(nonaller_sample)*0.8): \
        int(len(nonaller_sample)*0.9)]
nonaller_val = nonaller_sample[int(len(nonaller_sample)*0.9):]

with open("data.log", 'w') as datalog:
    datalog.write("[ Trainings set ]\n")
    datalog.write("[ Allergens ]\n")
    datalog.writelines([i[0]+'\n' for i in aller_train])
    datalog.write("[ Non-Allergens ]\n")
    datalog.writelines([i[0]+'\n' for i in nonaller_train])

    datalog.write("[ Test set ]\n")
    datalog.write("[ Allergens ]\n")
    datalog.writelines([i[0]+'\n' for i in aller_test])
    datalog.write("[ Non-Allergens ]\n")
    datalog.writelines([i[0]+'\n' for i in nonaller_test])

    datalog.write("[ Validation set ]\n")
    datalog.write("[ Allergens ]\n")
    datalog.writelines([i[0]+'\n' for i in aller_val])
    datalog.write("[ Non-Allergens ]\n")
    datalog.writelines([i[0]+'\n' for i in nonaller_val])
    datalog.close()

oh_aller_train = np.array(
        list(map(onehot, list([i[0] for i in aller_train])))
        )
oh_aller_test = np.array(
        list(map(onehot, list([i[0] for i in aller_test])))
        )
oh_aller_val = np.array(
        list(map(onehot, list([i[0] for i in aller_val])))
        )

oh_nonaller_train = np.array(
        list(map(onehot, list([i[0] for i in nonaller_train])))
        )
oh_nonaller_test = np.array(
        list(map(onehot, list([i[0] for i in nonaller_test])))
        )
oh_nonaller_val = np.array(
        list(map(onehot, list([i[0] for i in nonaller_val])))
        )

rows = longest
n = 256 # Number of filters
inpt = Input(shape=(rows, 20))
rshp = Reshape(target_shape=(rows, 20, 1))(inpt)
conv2d = Conv2D(filters=n, activation='relu', kernel_size=(20, 1), padding='same')(rshp)
batchnorm = BatchNormalization()(conv2d)
rshp_1d = Reshape(target_shape=(rows*20, n))(batchnorm)

rb = resblock(rshp_1d, n)

for i in range(7):
    rb = resblock(rb, n)
    rb = resblock(rb, n, shortcut=True)
    n = int(n/2)

rb = resblock(rb, 2)
rb = resblock(rb, 1, shortcut=True)

flat = Flatten()(rb)
dense = Dense(100, activation='relu')(flat)
output = Dense(2, activation='softmax')(dense)

model = Model(inpt, output)
model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=['accuracy']
        )

epochs = 12
batch_size = 8

x_train = np.concatenate([oh_aller_train, oh_nonaller_train])
y_train = np.concatenate([oh_aller_test, oh_nonaller_test])
x_label = tf.keras.utils.to_categorical(
        np.concatenate([np.ones(1653), np.zeros(1653)]), 2
        )
y_label = tf.keras.utils.to_categorical(
        np.concatenate([np.ones(207), np.zeros(207)]), 2
        )

model.fit(
        x_train, x_label,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(y_train, y_label)
        )

score = model.evaluate(x_test, y_test)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
model.save("resnet30.hd5")
