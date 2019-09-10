import os
import datetime
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, Dense, Flatten, Reshape, Conv1D, Conv2D, Add, BatchNormalization, MaxPooling1D)
import numpy as np
from random import sample, shuffle
from math import ceil


def resblock(input_, filters_, shortcut=False):
    """Creates a Residual Layer with x amount of filters. Returns the bottom
    layer.
    """
    conv1d_r1 = Conv1D(
                    filters=filters_,
                    kernel_size=3,
                    padding="same",
                    activation='relu',
                    kernel_initializer='he_normal',
                )(input_)
    conv1d_r2 = Conv1D(
                    filters=filters_,
                    kernel_size=3,
                    padding="same",
                    activation='relu',
                    kernel_initializer='he_normal',
                )(conv1d_r1)
    conv1d_r3 = Conv1D(
                    filters=filters_,
                    kernel_size=1,
                    padding="same",
                    activation='relu',
                    kernel_initializer='he_normal',
                )(conv1d_r2)
    maxpool_r = MaxPooling1D(pool_size=2)(conv1d_r3)

    maxpool_r = BatchNormalization()(maxpool_r)

    if shortcut:
        conv1d_l = Conv1D(filters=filters_, kernel_size=3, padding="same", kernel_initializer='he_normal')(input_)
        maxpool_l = MaxPooling1D(pool_size=2)(conv1d_l)
        maxpool_l = BatchNormalization()(maxpool_l)

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

longest = 16384
epochs = 50
batch_size = 32


def numberOfLines(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def read_file_training(filename):
    with open(filename, "r") as f:
        while True:
            d = f.readline().replace('\n', '')
            
            if d == '': break
            
            d = d.split(";")
        
            yield onehot(d[0]), 1.0 if d[1] == '1' else 0.0

def read_file_validation(filename):
    with open(filename, "r") as f:
        while True:
            d = f.readline().replace('\n', '')
            
            if d == '': break
            
            d = d.split(";")
        
            yield onehot(d[0]), 1.0 if d[1] == '1' else 0.0


def get_model(filters=128, depth=3):
    """
    Create a residual neural network with a number of filters and
    a specified depth.

    The resulting network will have 2*depth many residual blocks 
    plus the layers from the embedding and the ouput.

    Args:
        filters     The number of filters (default: 128)
        depth       The depth of the residual network
    """

    # Input
    inputLayer = Input(shape=(longest, 20))
    reshapedLayer = Reshape(target_shape=(longest, 20, 1))(inputLayer)

    # Embedding layer: from amino acid sequence to embedded sequence
    embeddingLayer = Conv2D(filters=filters, activation='relu', kernel_size=(1, 20), kernel_initializer='he_normal')(reshapedLayer)
    embeddingLayer = BatchNormalization()(embeddingLayer)
    embeddingLayer = Reshape(target_shape=(longest, filters))(embeddingLayer)

    # Start the residual blocks
    residualLayers = resblock(embeddingLayer, filters)

    # Append the new residual blocks
    for i in range(depth-1):
        residualLayers = resblock(residualLayers, filters)
        residualLayers = resblock(residualLayers, filters, shortcut=True)
        filters = int(filters/2)

    # Append the final block
    residualLayers = resblock(residualLayers, 1, shortcut=True)

    # Flatten
    flattenedLayer = Flatten()(residualLayers)

    # Add dropout
    flattenedLayer = tf.keras.layers.Dropout(0.2)(flattenedLayer)

    # Get the output layer
    outputLayer = Dense(1, activation='sigmoid')(flattenedLayer)

    # Build the model and return
    model = Model(inputLayer, outputLayer)
    model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
    )

    return model


def train(split, timestamp, name="network", filters=128, depth=3, epochs=50, fromScratch=False, modelFilename=None):
    trainFilename = F".\data\splitted\data_split_{split}_train_{timestamp}.dat"
    validFilename = F".\data\splitted\data_split_{split}_valid_{timestamp}.dat"

    training_ds = tf.data.Dataset.from_generator(read_file_training, args=[trainFilename], output_types=(tf.int32, tf.float32), output_shapes=((longest, 20), ()))
    training_ds = training_ds.repeat().batch(batch_size)

    numDataPointsTrain = numberOfLines(trainFilename)
    numDataPointsValid = numberOfLines(validFilename)

    validation_ds = tf.data.Dataset.from_generator(read_file_validation, args=[validFilename], output_types=(tf.int32, tf.float32), output_shapes=((longest, 20), ()))
    validation_ds = validation_ds.repeat().batch(batch_size)

    # Get the model
    if not fromScratch and len(modelFilename) > 0:
        model = tf.keras.models.load_model(modelFilename)
    else:
        model = get_model(filters=filters, depth=depth)

    # Filename to store the models
    the_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = ".\\networks\\{}_{}.ckpt".format(name, the_timestamp)

    # Create callbacks
    checkPointCallback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_best_only=True,
                                                 verbose=1)
    loggingCallback = tf.keras.callbacks.CSVLogger(F'.\\networks\\training_{the_timestamp}.log')
    
    # Train the model, with checkpoints and storing the history of each epoch
    model.fit(
        training_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[loggingCallback, checkPointCallback],
        steps_per_epoch=ceil(numDataPointsTrain / batch_size),
        validation_data=validation_ds,
        validation_steps=ceil(numDataPointsValid / batch_size)
    )


if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split',
        type=int,
        default=1
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        required=True
    )
    parser.add_argument(
        '--filters',
        type=int,
        default=128
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=3
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50
    )
    parser.add_argument(
        '--model',
        type=str,
    )
    parser.add_argument(
        '--name',
        type=str,
        default="network"
    )

    # Parse the arguments
    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print("Error: Unrecognized options: {unparsed}")
        exit(-1)
    
    # Do the training
    train(
        FLAGS.split, 
        FLAGS.timestamp, 
        name=FLAGS.name,
        filters=FLAGS.filters,
        depth=FLAGS.depth,
        epochs=FLAGS.epochs,
        fromScratch=FLAGS.model is None,
        modelFilename=FLAGS.model
    )
    
