from Bio import SeqIO, pairwise2, AlignIO
import sys

def seq_id(seq1, seq2):
    """Takes two sequences as input and calculates their similarity based on
    the alignment score divided by the alignment length
    """
    alignment = pairwise2.align.globalxx(seq1, seq2)[0]
    
    return alignment[-3] / alignment[-1]


flags = ['-a', '-n', '-h']
passed = [f for f in sys.argv if '-' in f]

if len([i for i in passed if i not in flags]) > 0:
    raise Exception("Unrecognized Flags")

if '-h' in sys.argv:
    print("Use -a to pass a .fasta file with allergens and -n with nonallergens")
    sys.exit()

allergens = list(SeqIO.parse(sys.argv[sys.argv.index('-a')+1], 'fasta'))

for seq in range(len(allergens)):

    if len(allergens[seq].seq) < 100:
        allergens.pop(seq)

    if 'X' in allergens[seq].seq:
        allergens.pop(seq)

nonallergens = list(SeqIO.parse(sys.argv[sys.argv.index('-n')+1], 'fasta'))

for seq in range(len(allergens)):

    if len(allergens[seq].seq) < 100:
        allergens.pop(seq)

    if 'X' in allergens[seq].seq:
        allergens.pop(seq)

for seq1 in range(len(allergens)):
    to_pop = []

    for seq2 in range(len(nonallergens)):
        identity = seq_id(allergens[seq1].seq, nonallergens[seq2].seq)

        if identity >= 0.9:
            to_pop.append(seq2)

    to_pop = list(dict.fromkeys(to_pop))
    to_pop = sorted(to_pop, reverse=True)

    for p in to_pop:
        nonallergens.pop(p)

filtered = open("filtered.fasta", 'w')

for seq in range(len(allergens)):
    filtered.write(str(allergens[seq].seq)+";1\n")

for seq in range(len(nonallergens)):
    filtered.write(str(nonallergens[seq].seq)+";0\n")

filtered.close()
