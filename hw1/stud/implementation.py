

import numpy as np
from typing import List, Tuple
import string

from typing import List, Dict
import json
import torch
import random
import csv
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

from model import Model


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
print(device,"eeeeeeeeeeeeeeeee")
#setting unknown token  to handle out of vocabulary words
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
TRAIN_PATH = "./data/train.tsv"
DEV_PATH = "./data/dev.tsv"



#creating a vocabulary with glove embeddings
def create_glove(embedding_dim=EMBEDDING_DIM):
    f = open('./hw1/stud/glove/glove.6B.' + str(embedding_dim) + 'd.txt', 'r')
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
    word2idx[PAD_TOKEN] = 1
    for word,vector in vocabulary.items():      #creating the word:index entry and insert in vectors
        word2idx[word] = len(vectors)           #the word vector at the corresponding index for each word
        vectors.append(torch.tensor(vector))    #in the dictionary
    word2idx = defaultdict(lambda: 0, word2idx) #if the word we're looking for is not in the dictionary, we give the unknown token index
    vectors = torch.stack(vectors).to(device).float()   #convert the list of tensors into a tensor of tensors
    return vectors,word2idx

glove = create_glove()  # create glove dictionary
embeddings, word2idx = create_embeddings(glove)  # create and indexing embeddings


class SentenceDataset(Dataset):

    def __init__(self,vectors,word2idx,sentences_path=None,sentences=None,lemmatization=True,test=False):
        file_output = self.read_file(sentences_path) if sentences_path else self.read_sentences(sentences)
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
        #print(sentences)
        return sentences

    def read_sentences(self,sentences):
        sents = list()
        for idx,line in enumerate(sentences):
            d = dict()
            d["id"] = idx
            d["text"] = line
            d["labels"] = ["O" for token in line]
            sents.append(d)
        return sents

    #function to extract the sentences from the dictionary of samples
    def extract_sentences(self,file_output):
        self.sentences = list()                 #creating a list to store the instances in the dataset
        for instance in file_output:
            processed = self.text_preprocess(instance)  #preprocessing of the sentence
            labels = 'UNKNOWN'   #this is needed to make the system able to give a prediction without having a ground truth
            if 'labels' in instance: #then if there is a ground truth we take it
                labels = processed['labels']
            self.sentences.append((processed["text"], labels, id))           #append a triple (sentence,label,id) which are all the informations we need
        if not self.test: random.Random(42).shuffle(self.sentences)         #for the training phase, shuffle data to avoid bias relative to data order
        #print(self.sentences)

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
        text = sentence["text"]
        labels = sentence["labels"]
        sent = [(text[i],labels[i]) for i in range(len(text))]# if text[i] not in string.punctuation and text[i] not in stop_tokens]
        sentence["text"] = [pair[0] for pair in sent]
        sentence["labels"] = [pair[1] for pair in sent]
        return sentence
        '''
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
        '''




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
        #return torch.tensor([word2idx[word] for word,pos in sent]) #in the case i'm not using pos I just return a placeholder for pos
        return torch.tensor([word2idx[word] for word in sent])



    #custom collate function, used to create the batches to give as input to the nn
    #it's needed because we are dealing with sentences of variable length and we need padding
    #to be sure that each sentence in a batch has the same length, which is necessary since
    #neural networks need fixed dimension inputs
    def collate(self,word2idx, data):
        X = [self.sent2idx(instance[0], word2idx) for instance in data]                             #extracting the input sentence
        X_len = torch.tensor([x.size(0) for x in X], dtype=torch.long).to(device)
        y = [self.sent2idx(instance[1], self.class2id) for instance in data]
        ids = [instance[2] for instance in data]                                                    #extracting the sentence ids
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=1).to(device)        #padding all the sentences to the maximum length in the batch (forcefully max_len)
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=13).to(device)              #extracting the ground truth
        return X, X_len,y, ids

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates

    return StudentModel(embeddings=embeddings)


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # constructor method, for which are needed the word embedding vectors, the dimensions of the two linear
    # layers, the dropout probabilty p, a flag to choose if the lstm layer must be bidirectonal, the number
    # of layers of the lstm layer and the loss function (but these last 4 already have a default value)
    def __init__(self, embeddings,  # word embedding vectors
                 pos_embeddings=None,  # pos embedding vectors
                 hidden1=512,  # dimension of the first hidden layer
                 hidden2=512,  # dimension of the second hidden layer
                 p=0.0,  # probability of dropour layer
                 bidirectional=False,  # flag to decide if the LSTM must be bidirectional
                 lstm_layers=1,  # layers of the LSTM
                 loss_fn=torch.nn.CrossEntropyLoss(ignore_index=13)):  # loss function
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.pos_embeddings = None if pos_embeddings == None else nn.Embedding.from_pretrained(pos_embeddings,
                                                                                               freeze=False)  # choose if creating or not an embedding layer
        hidden1 = hidden1 * 2 if bidirectional else hidden1  # based on wether pos embeddings were created
        input_dim = embeddings.size(1) if not self.pos_embeddings else embeddings.size(1) + pos_embeddings.size(
            1)  # or not
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden1, num_layers=lstm_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.lin1 = nn.Linear(hidden1, 13)
        # self.lin2 = nn.Linear(hidden2,13)
        self.loss_fn = loss_fn
        self.dropout = nn.Dropout(p=p)

    # forward method, automatically called when calling the instance
    # it takes the Xs and their length in batches
    def forward(self, X, X_len):
        # lemmas,pos = X[...,0],X[...,1]                                  #separating pos from lemmas
        embedding = self.embedding(X)  # expanding the words from indices to embedding vectors
        '''
        if self.pos_embeddings != None:                                     #in the case I'm using pos embeddings, I pass their indexes through their own embedding layer
            pos_embeddings_out1 = self.pos_embeddings(pos)                 #and then concatenate them to the corresponding words

            embedding1 = torch.cat([embedding,pos_embeddings_out1],dim=-1)
        '''
        lstm_out = self.lstm(embedding)[0]
        batch_size, sentence_len, hidden_size = lstm_out.shape  # sentence length here is taken to remove padding
        flattened1 = lstm_out.reshape(-1,
                                      hidden_size)  # the output of the lstm is flattened (batch size,sentence length,lstm output dimension)->
        #                                       (batch size,sentence length*lstm output dimension)
        last_word_relative_indices = X_len - 1  # in order to have the last index of each sentence in the batch (excluding padding), we subtract one to the length
        sentences_offsets = torch.arange(batch_size,
                                         device=device) * sentence_len  # we compute the offsets, or absolute indexes, where each sentence starts
        vec_sum_index = sentences_offsets + last_word_relative_indices  # we take the last token of each sentence, which is the hidden
        # state of the LSTM which summarize the content of the sentence
        # vec_sum = flattened1[vec_sum_index]                         #we use the index to retrieve the vector which summarizes the sentence
        ##out = self.dropout(torch.cat([vec_sum1,vec_sum2],dim=-1))
        out = self.dropout(flattened1)
        out = torch.relu(out)
        # print(out.size())
        out = self.lin1(out)

        # out = self.lin2(out)
        out = out.squeeze(1)
        # print(out.size())
        out = torch.softmax(out, dim=-1)
        return out



    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        batch_size = 32
        predictions = list()
        dataset = SentenceDataset(sentences=tokens, vectors=embeddings, word2idx=word2idx,test=True)
        dataloader = dataset.dataloader(batch_size)
        for batch in dataloader:
            batch_x = batch[0]
            batch_xlen = batch[1]
            ids = batch[3]
            logits = self.forward(batch_x, batch_xlen)
            logits = logits.view(-1, logits.shape[-1])
            preds = torch.argmax(logits,dim=1)
            preds = torch.reshape(preds,(batch_x.size(0),-1))
            for i in range(len(batch_x)):
                prediction = []
                for j in range(len(batch_x[i])):
                    if batch_x[i][j].item() != 1:
                        #print(preds)
                        #print(preds[i])
                        prediction.append(dataset.id2class[preds[i][j].item()])
                predictions.append(prediction)



        return predictions
