import re
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Fancy stuff
sns.set()
from matplotlib import font_manager
font_manager._rebuild()
plt.rcParams['font.sans-serif']= "Open Sans"
plt.rcParams['font.weight'] = 'light'


##
# Helper functions

def onehot(seq):
    """Return the amino acid sequence as one hot coded numpy array"""
    oh = np.zeros([longest, 20])

    for i, j in enumerate(seq):
        oh[i][aa2int[j]] = 1

    return oh


aa1 = list("ACDEFGHIKLMNPQRSTVWY")
aa2int = dict((c, i) for i, c in enumerate(aa1))
longest = 16384

def numberOfLines(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def read_file(filename):
    with open(filename, "r") as f:
        while True:
            d = f.readline().replace('\n', '')
            
            if d == '': break
            
            d = d.split(";")
        
            yield onehot(d[0]), 1.0 if d[1] == '1' else 0.0


def decode_network_filename(filename):
    m = re.match('network_([0-9]{8})-([0-9]{6}).([A-Za-z0-9]*)', filename)
    if not m: return None, None

    return m.group(1), m.group(2)


##
# Analysis methods

def confusion_matrix(Xs, Ys_real, Ys_pred, threshold=0.0, channel=0):
    # Convert the predicted values into a one-hot-encoding
    predicted = (Ys_pred > threshold).transpose()
    real = Ys_real > threshold
    
    not_predicted = np.logical_not(predicted)
    not_real = np.logical_not(real)
    
    TP = np.logical_and(predicted, real).sum()
    TN = np.logical_and(not_predicted, not_real).sum()
    FP = np.logical_and(predicted, not_real).sum()
    FN = np.logical_and(not_predicted, real).sum()
    
    return TP, TN, FP, FN

def rates(Xs, Ys_real, Ys_pred, threshold=0.0, channel=0):
    TP, TN, FP, FN = confusion_matrix(Xs, Ys_real, Ys_pred, threshold=threshold, channel=channel)        
    return TP / (TP + FN), TN / (TN + FP), FP / (FP + TN), FN / (FN + TP)

def roc_curve(Xs, Ys_real, Ys_pred, numThresholds=100, channel=0):
    curve = { 0.0: 0.0, 1.0: 1.0 }
    for thresh in np.linspace(0.0, 1.0, numThresholds):
        tpr, _, fpr, _ = rates(Xs, Ys_real, Ys_pred, threshold=thresh, channel=channel)
        curve[fpr] = tpr
    
    # Sort the curve by the FPR values
    sorted_curve = list(sorted(list(curve.items()), key=lambda x : x[0]))
    
    # Unzip the curve and return the result
    return tuple(zip(*sorted_curve))

def pr_curve(Xs, Ys_real, Ys_pred, numThresholds=100):
    curve = {}
    for thresh in np.linspace(0.0, 1.0, numThresholds):
        TP, TN, FP, FN = confusion_matrix(Xs, Ys_real, Ys_pred, threshold=thresh)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        curve[precision] = recall
    
    # Sort the curve by the FPR values
    sorted_curve = list(sorted(list(curve.items()), key=lambda x : x[0]))
    
    # Unzip the curve and return the result
    return tuple(zip(*sorted_curve))

def best_threshold(Xs, Ys_real, Ys_pred, numThresholds=100):
    best = 0.0
    lastY = 0.0
    
    vs = {}
    
    for thresh in np.linspace(0.0, 1.0, numThresholds):
        tpr, _, fpr, _ = rates(Xs, Ys_real, Ys_pred, threshold=thresh)
        y = tpr + fpr - 1
        vs[thresh] = abs(y)        
        
    return min(vs, key=vs.get)

def area_under_curve(x, y):
    a = 0.0
    for i in range(len(x)-1):
        dx = x[i+1]-x[i]
        v = y[i+1]+y[i]
        if np.isnan(dx) or np.isnan(v): continue
        a += v*dx / 2.0
    return a


##
# Plots

def history(acc, loss, filename, output):
    # Nothing to plot? Stop
    if not acc and not loss: return

    print("---  Plot history")

    filename = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace("network", "training").replace(".ckpt", ".log"))
    history = pd.read_csv(filename, sep=',')

    # Start plot
    plt.figure(figsize=(10,8))
    plt.suptitle("History\n")

    index = 1
    
    # Plot accuracy
    if acc:
        plt.subplot(2 if acc and loss else 1, 1, index)
        plt.plot(history['epoch'], history['acc'], label='Training')
        plt.plot(history['epoch'], history['val_acc'], label='Validation')
        plt.title('Accuracy\n')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.xlim(0, max(history['epoch']))
        plt.legend()
        index += 1

    # Plot loss 
    if loss:
        plt.subplot(2 if acc and loss else 1, 1, index)
        plt.plot(history['epoch'], history['loss'], label='Training')
        plt.plot(history['epoch'], history['val_loss'], label='Validation')
        plt.title('Loss\n')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, max(history['epoch']))
        plt.legend()
        index += 1

    # Save figure
    plt.savefig(output + ".png")


def plot_roc_curve(X, Y, Ypred, output):
    plt.figure(figsize=(15,8))
    plt.title("Receiver Operator Characteristic\n")

    curve = roc_curve(X, Y, Ypred, numThresholds=1000)
    auc = area_under_curve(*curve)

    # Write to text file
    df = pd.DataFrame({'x': curve[0], 'y': curve[1], 'auc': auc })
    df.to_csv(output + '.dat')
    
    # Start the plot
    plt.plot(curve[0], curve[1], linewidth=2.0)
    
    the_labels = [F"Allergen-net ({auc:.2f})", "Random classifier (0.5)"]
    the_labels.append("Random classifier (0.5)")
    
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend(the_labels)

    #plot_roc_curve(X, Y, Ypred, numThresholds=1000)
    plt.savefig(output + '.png')


def plot_pr_curve(X, Y, Ypred, output):
    plt.figure(figsize=(15,8))
    plt.title("Precision Recall")

    curve = pr_curve(X, Y, Ypred, numThresholds=1000)

    # Write to text file
    df = pd.DataFrame({'x': curve[0], 'y': curve[1] })
    df.to_csv(output + '.dat')
    
    # Start the plot
    plt.plot(curve[0], curve[1], linewidth=2.0)
    
    plt.xlim(0,1)
    plt.ylim(0,1)

    #plot_roc_curve(X, Y, Ypred, numThresholds=1000)
    plt.savefig(output + '.png')

##
# The analysis

def analysis(**kwargs):
    # Load the model and the data
    modelFilename = kwargs.get('model')
    testFilename = kwargs.get('test')

    # Get the timestamp
    timestamp = "-".join(decode_network_filename(os.path.basename(modelFilename)))

    # Plot the history
    history(kwargs.get('acc'), kwargs.get('loss'), modelFilename, os.path.join(kwargs.get('output'), F"history_{timestamp}"))

    # Load the data
    print("---  Load test data")
    numDataPoints = numberOfLines(testFilename)
    data = [d for d in read_file(testFilename)]

    # Transform the data
    print("---  Transform data")
    X = np.zeros((len(data),) + data[0][0].shape)
    Y = np.zeros((len(data),))
    for i in range(numDataPoints):
        X[i] = data[i][0]
        Y[i] = data[i][1]

    # Load the model
    print("--- Load model") 
    model = tf.keras.models.load_model(modelFilename)

    # Predict
    print("--- Predict labels for test data") 
    Ypred = model.predict(X, batch_size=2)

    # Do the receiver operator curve
    if kwargs.get('roc'):
        plot_roc_curve(X, Y, Ypred, os.path.join(kwargs.get('output'), F"roc_{timestamp}"))
    
    # Do the precision recall curve
    if kwargs.get('pr'):
        plot_pr_curve(X, Y, Ypred, os.path.join(kwargs.get('output'), F"pr_{timestamp}"))

    print("---  Finished.")


##
# CLI

def main():
    parser = argparse.ArgumentParser()

    # Flags for the different metrics
    parser.add_argument('--roc', action='store_true')
    parser.add_argument('--pr', action='store_true')
    parser.add_argument('--acc', action='store_true')
    parser.add_argument('--loss', action='store_true')

    # Model and test data set
    parser.add_argument(
        '--test',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./'
    )

    # Parse
    args, unparsed = parser.parse_known_args()

    if unparsed:
        print("Error: Unrecognized options: {unparsed}")
        exit(-1)
    
    # Do the analysis
    analysis(**vars(args))


if __name__ == '__main__':
    main()
