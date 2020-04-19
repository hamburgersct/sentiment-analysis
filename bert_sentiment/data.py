# define a configurable SSTDataset class

import pytreebank
import torch
from loguru import logger
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer

logger.info('Loading the tokenizer...')
# use Bert-large
vocab_dir = './vocab'
vocab_txt_name = '/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_dir + vocab_txt_name)

logger.info('Loading SST...')
sst = pytreebank.load_sst()

def rpad(array, n=70):
    """Right padding"""
    current_len = len(array)
    if current_len > n:
        return array[:n-1]
    empty = n - current_len
    return array + ([0] * empty)

"""
    labeled from: 0 -> very negative
                  2 -> neural
                  4 -> very positive
"""
def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label < 2:
        return 0
    if label > 2:
        return 1
    raise ValueError('Invalid label')

class SSTDataset(Dataset):
    """ configurable SST Dataset

        configuration:
            - split: train/val/test
            - nodes: root/all
            - binary/fine-grained
    """
    def __init__(self, split='train', root=True, binary=True):
        """Initializes the dataset with given configuration.
            - split: str
                Dataset split, [train, val, test]
            - root: bool
                if true, only use root nodes; else, use all nodes
            - binary: bool
                if true, use binary labels; else, use fine-grained
        """
        logger.info(f"Loading SST {split} set")
        self.sst = sst[split]

        logger.info("Tokenizing")
        if root and binary:
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS]" + tree.to_lines()[0] + "[SEP]", n = 66)
                    ),
                    get_binary_label(tree.label),
                )
                for tree in self.sst
                if tree.label != 2
            ]
        elif root and not binary:
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS]" + tree.to_lines()[0] + "[SEP]"),
                        n = 66
                    ),
                    tree.label,
                )
                for tree in self.sst
            ]
        elif not root and not binary:
            self.data = [
                (rpad(tokenizer.encode("[CLS]" + line + "[SEP]"), n = 66), label)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
            ]
        else:
            self.data = [
                (
                    rpad(tokenizer.encode("[CLS]" + line + "[SEP]"), n = 66),
                    get_binary_label(label),
                )
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        X = torch.tensor(X)
        return X,y