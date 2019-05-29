import numpy as np
import theano
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import csv
import gensim

def build_data_cv(datafile, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)

    with open(datafile, "r") as csvf: #encoding = 'cp1252'
        csvreader=csv.reader(csvf,delimiter=',',quotechar='"')
        first_line=True
        for line in csvreader:
            if first_line:
                first_line=False
                continue
            status=[]
            sentences=re.split(r'[.?]', line[1].strip())
            try:
                sentences.remove('')
            except ValueError:
                None

            for sent in sentences:
                if clean_string:
                    orig_rev = clean_str(sent.strip())
                    if orig_rev=='':
                            continue
                    words = set(orig_rev.split())
                    splitted = orig_rev.split()
                    if len(splitted)>150:
                        orig_rev=[]
                        splits=int(np.floor(len(splitted)/20))
                        for index in range(splits):
                            orig_rev.append(' '.join(splitted[index*20:(index+1)*20]))
                        if len(splitted)>splits*20:
                            orig_rev.append(' '.join(splitted[splits*20:]))
                        status.extend(orig_rev)
                    else:
                        status.append(orig_rev)
                else:
                    orig_rev = sent.strip().lower()
                    words = set(orig_rev.split())
                    status.append(orig_rev)

                for word in words:
                    vocab[word] += 1


            datum  = {"y0":1 if line[2].lower()=='y' else 0,
                  "y1":1 if line[3].lower()=='y' else 0,
                  "y2":1 if line[4].lower()=='y' else 0,
                  "y3":1 if line[5].lower()=='y' else 0,
                  "y4":1 if line[6].lower()=='y' else 0,
                  "text": status,
                  "user": line[0],
                  "num_words": np.max([len(sent.split()) for sent in status]),
                  "split": np.random.randint(0,cv)}
            revs.append(datum)


    return revs, vocab

def get_W(model_vocab, vocab, w2v, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(model_vocab)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype=theano.config.floatX)
    W[0] = np.zeros(k, dtype=theano.config.floatX)
    i = 1
    for word in model_vocab:
        W[i] = w2v[word]
        word_idx_map[word] = i
        i += 1
        
        #Printing how many words left to complete the matrix:
        if i%500000 == 0:
            print('Words Appended :' + str(i) + '/3,000,000')
    return W, word_idx_map

def add_unknown_words(w2v, model_vocab, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    l = len(vocab)
    for j, word in enumerate(vocab):
        if word not in model_vocab and vocab[word] >= min_df:
            w2v.vocab[word] = np.random.uniform(-0.25,0.25,k)
        
        #Printing to see progress
        if j%(l/20) == 0:
            print('Words Appended :' + str(j) + '/' + str(l))
        

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d" , " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
#    string = re.sub(r"[a-zA-Z]{4,}", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_mairesse_features(file_name):
    feats={}
    with open(file_name, "rb") as csvf:
        csvreader=csv.reader(csvf,delimiter=',',quotechar='"')
        for line in csvreader:
            feats[line[0]]=[float(f) for f in line[1:]]
    return feats

def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert_keys_to_string(v)) 
        for k, v in dictionary.items())


if __name__=="__main__":
    #data_folder = sys.argv[1]
    data_folder = 'essays.csv'
    #mairesse_file = sys.argv[2]
    mairesse_file = 'mairesse.csv'
    print "loading data...",
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    vocab = convert_keys_to_string(vocab)
    num_words=pd.DataFrame(revs)["num_words"]
    max_l = np.max(num_words)
    print "data loaded!"
    print "number of status: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors..."

    #Loading Word2Vec from gensim model
    w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print "word2vec loaded!"
    model_vocab = w2v.vocab.keys()
    print "num words already in word2vec: " + str(len(model_vocab))
    #%%
    #Adding unknown words from the vocab into the word vector
    add_unknown_words(w2v, model_vocab, vocab)
    #%%
    #Getting the 300xlen(vocab) word matrix
    W, word_idx_map = get_W(model_vocab, vocab, w2v)
    mairesse = get_mairesse_features(mairesse_file)
    cPickle.dump([revs, W, W2, word_idx_map, vocab, mairesse], open("essays_mairesse.p", "wb"))
    print "dataset created!"

