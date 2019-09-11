import numpy as np
from random import sample, shuffle

aa1 = list("ACDEFGHIKLMNPQRSTVWY")
aa2int = dict((c, i) for i, c in enumerate(aa1))

#Parsing of filtered dataset file
raw = list(open("parsed.fasta", 'r'))
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

def onehot(seq):
    """Return the amino acid sequence as one hot coded numpy array"""
    oh = np.zeros([longest, 20])

    for i, j in enumerate(seq):
        oh[i][aa2int[j]] = 1

    return oh


oh_aller_train = [list(map(onehot, list([i[0] for i in aller_train])))]
oh_aller_test = [list(map(onehot, list([i[0] for i in aller_test])))]
oh_aller_val = [list(map(onehot, list([i[0] for i in aller_val])))]

oh_nonaller_train = [list(map(onehot, list([i[0] for i in nonaller_train])))]
oh_nonaller_test = [list(map(onehot, list([i[0] for i in nonaller_test])))]
oh_nonaller_val = [list(map(onehot, list([i[0] for i in nonaller_val])))]
