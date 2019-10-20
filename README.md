# allerGEM

allergen-net allows to predict if a protein sequence is an allergen. The project is created as part of
the annual iGEM competition 2019 by the team of the Friedrich-Alexander-University Erlangen-Nuremberg.

## Usage

### Training

```sh
python train.py --timestamp "20190807-102903" --filters=128 --depth=4 --epochs 150
```

The `--timestamp` option determines which datasplit to use (standard split is `1`, but one can switch to another one via `--split`).
Normally the network trains from scratch, unless a `--model=<FILENAME>` option is supplied, then it continues training with the checkpoint.

### Analysis

```sh
python analyse.py --test <path-to-test-set> --model <path-to-network> --acc --loss --roc --pr
```

Depending on the options, the history, the receiver operator characteristics and the precision recall curve are plotted to a file in the current directory. If `--output=<path>` is supplied, then the plots are stored in the given directory.
