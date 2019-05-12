from Bio import SeqIO, pairwise2, AlignIO
import sys

helptxt = "Filters a .fasta file for protein sequences longer than 100 amino \
acids and containing only standard amino acids, i.e. no \"X\".\nAdditionally \
sequences that have a over 90% sequence identity will be deleted"

if sys.argv[1] == '-h':
    print(helptxt)
    sys.exit()


def seq_id(seq1, seq2):
    """Takes two sequences as input and calculates their similarity based on
    the alignment score divided by the alignment length
    """
    alignment = pairwise2.align.globalxx(seq1, seq2)[0]
    
    return alignment[-3] / alignment[-1]


def write_fasta(fastalst, handle):
    """Writes list(SeqIO) object to a file.
    """
    for seq in fastalst:
        handle.write('>')
        handle.write(seq.description)
        handle.write('\n')
        handle.write(str(seq.seq))
        handle.write('\n')


sequences = list(SeqIO.parse(sys.argv[1], 'fasta'))[:10]

for seq in range(len(sequences)):
    
    if len(sequences[seq].seq) < 100 and 'X' in sequences[seq].seq:
        sequences.pop(seq)

to_pop = []
for seq1 in range(len(sequences)-1):

    for seq2 in range(seq1+1, len(sequences)):
        identity = seq_id(sequences[seq1].seq, sequences[seq2].seq)

        if identity >= 0.9:
            to_pop.append(seq2)

#Delete duplicates and sort the indexes in descending order so deleting them one
#by one won't mess up the rest of the indices.
to_pop = list(dict.fromkeys(to_pop))
to_pop = sorted(to_pop, reverse=True)

for p in to_pop:
    sequences.pop(p)

filtered = open("filtered.fasta", 'w')
write_fasta(sequences, filtered)
filtered.close()
