from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from torchtext.vocab import Vectors
import string

def tokenize(input):
    """
        Naive tokenizer, that lower-cases the input
        and splits on punctuation and whitespace
    """
    input = input.lower()
    for p in string.punctuation:
        input = input.replace(p," ")
    return input.strip().split()


def num2words(vocab,vec):
    """
        Converts a vector of word indicies
        to a list of strings
    """
    return [vocab.itos[i] for i in vec]

def get_imdb(batch_size,max_length):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True,tokenize=tokenize,fix_length=max_length)
    LABEL = data.Field(sequential=False,unk_token=None,pad_token=None)

    print("Loading IMDB data..\n")

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # print information about the data
    print(('train.fields', train.fields))
    print(('len(train)', len(train)))
    print(('len(test)', len(test)))
    print("")

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='42B', dim=300,max_vectors=500000))
    LABEL.build_vocab(train)

    # print vocab information
    print(('len(TEXT.vocab)', len(TEXT.vocab)))
    print(('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size()))

    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size)

    return train_iter, test_iter, TEXT.vocab.vectors, TEXT.vocab

def get_amazon(batch_size,max_length):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True,tokenize=tokenize,fix_length=max_length)
    LABEL = data.Field(sequential=False,unk_token=None,pad_token=None)

    print("Loading AMAZON data..\n")

    # make splits for data
    train, test = data.TabularDataset.splits(
        path='.data/amazon/', format='csv', skip_header=False,
        train='dataset_train.csv', test='dataset_test.csv',
        fields=[
            ('text', TEXT),
            ('label', LABEL),
        ])

    # print information about the data
    print(('train.fields', train.fields))
    print(('len(train)', len(train)))
    print(('len(test)', len(test)))
    print("")

    path='.vector_cache/'
    model_name='plainwordembeds'

    #load word2vec
    vectors = Vectors(name=model_name, cache=path)  # model_name + path = path_to_embeddings_file

    # build the vocabulary
    # provide the embedded vectors when you call build_vocab function
    TEXT.build_vocab(
        train,
        #max_size=self.config.vocab_maxsize,
        #min_freq=self.config.vocab_minfreq,
        vectors=vectors
    )
    LABEL.build_vocab(train)

    #set embedded vectors
    TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)

    # print vocab information
    print(('len(TEXT.vocab)', len(TEXT.vocab)))
    print(('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size()))

    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size, shuffle=True, repeat=False, sort=False)

    return train_iter, test_iter, TEXT.vocab.vectors, TEXT.vocab

if __name__ == "__main__":
    """
        If run seperately, does a simple sanity check,
        by printing different values,
        and counting labels
    """
    train, test, vectors, vocab = get_imdb(1,50)
    from collections import Counter
    print(list(enumerate(vocab.itos[:100])))
    cnt = Counter()
    for i,b in enumerate(iter(train)):
        if i > 2: break
        print(i,num2words(vocab,b.text[0][0].numpy()))
        cnt[b.label[0].item()] += 1
    print(cnt)
