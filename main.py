import warnings
warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize
import torch
from torchtext import data
from args import Args
from textCNN import textCNN
from train_model import train_model

arg = Args()
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def tokenizer(sentence_list):
    return [x for x in word_tokenize(sentence_list)]
TEXT = data.Field(sequential=True, tokenize=tokenizer,fix_length=250)
LABEL = data.Field(sequential=False, use_vocab=False)
print("preparing dataset...")
train, dev, test = data.TabularDataset.splits(path='dataset',
                                              train='train.csv',
                                              validation='dev.csv',
                                              test='test.csv',
                                              format='csv',
                                              skip_header=True,
                                              csv_reader_params={'delimiter':'\t'},
                                              fields=[('text',TEXT),('label',LABEL)])
print("building vocab...")
TEXT.build_vocab(train)

train_iter, val_iter, test_iter = data.Iterator.splits((train,dev,test),
                                                             batch_size = arg.batch_size,
                                                             sort_key=lambda x:len(x.text),
                                                             sort_within_batch=False,
                                                             repeat=False)

train_model(train_iter, val_iter, textCNN(arg), arg)