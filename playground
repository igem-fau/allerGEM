"""
iGEM FAU 2019
"""
import numpy as np

class FASTA:
    """
    Takes a .fasta file as an input and saves relevant information in its
    respective attributes.
    """
    def __init__(self, filename):
        """
        Attributes:
        -"file" is a list object of the opened file.
        -"aa_seq" returns the amino acid sequence.
        -"header" returns the header(s) of the file.
        -"comments" returns the comments (marked by ";")
        """
        self.file = list(open(filename, 'r'))
        self.aa_seq = ''.join([i.split('\n')[0] for i in self.file \
                if '>' and ';' not in i])
        self.header = ''.join([i.split('\n')[0] for i in self.file \
                if '>' in i[0]])
        self.comments = ''.join([i.split('\n')[0] for i in self.file \
                if ';' in i[0]])
    
        
    def __len__(self):
        """
        Returns the length of the amino acid sequence.
        """
        return len(self.aa_seq)


    def one_hot(self):
        """
        Turn AA sequence into a one hot coded version for neural networks.
        The numbers follow the order in the naming conventions.
        """
        aa1 = "ACDEFGHIKLMNPQRSTVWY"
        aa12num = [aa1.index(i) for i in self.aa_seq]
        coded = np.zeros([len(self.aa_seq), 20])

        for i in range(len(self.aa_seq)):
            idx = aa12num[i]
            coded[i, idx] = 1

        return coded