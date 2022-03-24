

import numpy as np
from typing import List, Tuple


from typing import List, Dict
import json
import torch
import random
import csv
import matplotlib.pyplot as plt
import nltk
from torch import nn
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from string import punctuation
from collections import defaultdict
from functools import partial



torch.manual_seed(42)
np.random.seed(42)
random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stop_tokens = set(stopwords.words('english'))
punc_tokens = set(punctuation)
stop_tokens.update(punc_tokens)
lemmatizer = WordNetLemmatizer()

#setting the embedding dimension
EMBEDDING_DIM=50

SENTENCE_MAX_LEN=50

#specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
#setting unknown token  to handle out of vocabulary words
UNK_TOKEN = '<unk>'

TRAIN_PATH = "./data/train.jsonl"
DEV_PATH = "./data/dev.jsonl"
TEST_PATH = "./data/test.jsonl"
DEV_PREDICTIONS_PATH = "./predictions/predictions_dev.tsv"
TEST_PREDICTIONS_PATH = "./predictions/predictions_test.tsv"



#creating a vocabulary with glove embeddings
def create_glove(embedding_dim=EMBEDDING_DIM):
    f = open('./glove/glove.6B.' + str(embedding_dim) + 'd.txt', 'r')
    glove = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        embedding = np.array([float(value) for value in splitLines[1:]])
        glove[word] = embedding
    return glove

#creating a vector of word embeddings and a dictionary to pair each word with an index
#the vector of embeddings will be needed in the embedding layer of the neural network
def create_embeddings(vocabulary,embedding_dim=EMBEDDING_DIM):
    vectors = []                                #creating a vector to append the vectors corresponding to words
    word2idx = dict()                           #creating a dictionary to associate each word with an index
    vectors.append(torch.rand(embedding_dim))   #creating a random vector for unknown (out of vocabulary) words
    vectors.append(torch.rand(embedding_dim))   #creating a random vector for padding
    word2idx[UNK_TOKEN] = 0                     #setting the index of the unknown token to 0
    for word,vector in vocabulary.items():      #creating the word:index entry and insert in vectors
        word2idx[word] = len(vectors)           #the word vector at the corresponding index for each word
        vectors.append(torch.tensor(vector))    #in the dictionary
    word2idx = defaultdict(lambda: 0, word2idx) #if the word we're looking for is not in the dictionary, we give the unknown token index
    vectors = torch.stack(vectors).to(device).float()   #convert the list of tensors into a tensor of tensors
    return vectors,word2idx

class SentenceDataset(Dataset):

    def __init__(self,sentences_path,vectors=0,word2idx=0,lemmatization=True,test=False):
        file_output = self.read_file(sentences_path)
        self.embedding_vectors = vectors
        self.word2idx = word2idx
        self.test = test
        self.w_lemmatization = lemmatization
        self.word_count = dict()
        self.extract_sentences(file_output)
        #self.remove_most_frequent(percentage=1)
        # encoding of classes
        self.class2id = {"O": 0, "B-PER": 1, "B-LOC": 2, "B-GRP": 3, "B-CORP": 4, "B-PROD": 5, #indexing output classes
                    "B-CW": 6, "I-PER": 7, "I-LOC": 8, "I-GRP": 9, "I-CORP": 10, "I-PROD": 11, "I-CW": 12}

        self.id2class = {v: k for (k, v) in self.class2id.items()}

    #function to remove the most frequent words in the corpus (ideally the less discriminative, only acting as noise, but
    #it didn't work great so I'm not sure
    def remove_most_frequent(self,percentage):
        #print("STARTING MOST FREQUENT")
        sorted_counts = sorted(self.word_count.items(),key=lambda x:x[1],reverse=True)  #sort dict with word counts
        total_words = len(sorted_counts)                                                #check the total number of words
        start_index = int((total_words/100)*percentage)                                 #compute the starting point to
        sorted_counts = sorted_counts[start_index:]                                     #exclude the wanted percentage
        sorted_counts = set([x[0] for x in sorted_counts])                              #transforming in set to make
        sentences = []                                                                  #checking the presence efficient (O(1))
        #print("STARTING REMOVING")
        for sample in self.sentences:
            sentence = sample[0]
            new_sentence = []
            for word,pos in sentence:
                if word in sorted_counts:
                    new_sentence.append((word,pos))
            sentences.append((new_sentence,sample[1],sample[2]))
        self.sentences = sentences


    #little function to read a file given the path
    def read_file(self,path):
        sentences = list()
        sentence = dict()
        with open(path) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for idx,line in enumerate(tsv_file):
                if len(line) > 0:
                    if line[0] == '#':
                        sentences.append(dict())
                        sentences[-1]["id"] = line[2]
                        sentences[-1]["text"] = []
                        sentences[-1]["labels"] = []
                    else:
                        sentences[-1]["text"].append(line[0])
                        sentences[-1]["labels"].append(line[1])
        print(sentences)
        return sentences

    #function to extract the sentences from the dictionary of samples
    def extract_sentences(self,file_output):
        self.sentences = list()                 #creating a list to store the instances in the dataset
        for instance in file_output:
            processed = self.text_preprocess(instance['text'])  #preprocessing of the sentence
            label = 'UNKNOWN'   #this is needed to make the system able to give a prediction without having a ground truth
            if 'labels' in instance: #then if there is a ground truth we take it
                label = instance['label']
            self.sentences.append((processed, label, instance["id"]))           #append a triple (sentence,label,id) which are all the informations we need
        if not self.test: random.Random(42).shuffle(self.sentences)         #for the training phase, shuffle data to avoid bias relative to data order

    #function to convert the pos extracted by nltk to the pos required by the very same library for lemmatization
    #I also use it to give pos='' to punctuation
    def get_standard(self,pos):
        if pos[0] == 'V': return wordnet.VERB
        if pos[0] == 'R': return wordnet.ADV
        if pos[0] == 'N': return wordnet.NOUN
        if pos[0] == 'J': return wordnet.ADJ
        return ''

    #function which includes all the preprocessing steps for the sentences, which are tokenization,
    #stopwords and punctuation removal,pos tagging and lemmatization
    def text_preprocess(self,sentence):
        tokenized = word_tokenize(sentence)
        tokens_n_pos = nltk.pos_tag(tokenized)
        standard_tokens = [(token,self.get_standard(pos)) for token,pos in tokens_n_pos]
        clean_standard = [(token,pos) for token,pos in standard_tokens if pos != '']   #light stopwords removal
        clean_standard2 = [(token, pos) for token, pos in standard_tokens if token not in stop_tokens]  # full stopwords removal
        if self.w_lemmatization:            #choosing if applying lemmatization
            lemmatized = [(lemmatizer.lemmatize(token.lower(),pos),pos) if pos != '' else (lemmatizer.lemmatize(token.lower()),'') for token,pos in clean_standard2]
            #print("STARTED BUILDING WORD COUNT")
            for lemma,pos in lemmatized:
                if lemma in self.word_count:
                    self.word_count[lemma] += 1
                else:
                    self.word_count[lemma] = 1
            return lemmatized
        return [(word,pos) for word,pos in clean_standard2]



    #function to return the number of instances contained in the dataset
    def __len__(self):
        return len(self.sentences)

    #function to get the i-th instance contained in the dataset
    def __getitem__(self, idx):
        return self.sentences[idx]

    #custom dataloader which incorporates the collate function
    def dataloader(self,batch_size):
        return DataLoader(self,batch_size=batch_size,collate_fn=partial(self.collate,self.word2idx))

        #function to map each lemma,pos in a sentence to their indexes
    def sent2idx(self ,sent, word2idx):
        return torch.tensor([word2idx[word] for word,pos in sent]) #in the case i'm not using pos I just return a placeholder for pos

    #custom collate function, used to create the batches to give as input to the nn
    #it's needed because we are dealing with sentences of variable length and we need padding
    #to be sure that each sentence in a batch has the same length, which is necessary since
    #neural networks need fixed dimension inputs
    def collate(self,word2idx, data):
        X = [self.sent2idx(instance[0], word2idx) for instance in data]                             #extracting the input sentence
        ids = [instance[2] for instance in data]                                                    #extracting the sentence ids
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=1).to(device)        #padding all the sentences to the maximum length in the batch (forcefully max_len)
        y = torch.tensor([self.class2id[instance[1]] for instance in data if instance[1] in self.class2id]).to(device)              #extracting the ground truth
        return X, ids, y


dataset = SentenceDataset(sentences_path="../../data/train.tsv")



