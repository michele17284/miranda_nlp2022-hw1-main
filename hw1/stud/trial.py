import string

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

TRAIN_PATH = "../../data/train.tsv"
DEV_PATH = "../../data/dev.tsv"



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
            d["labels"] = [-1 for token in line]
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
        sent = [(text[i],labels[i]) for i in range(len(text)) if text[i] not in string.punctuation and text[i] not in stop_tokens]
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


#Model class
#class StudentModel(nn.Module,Model):        #needed for testing
class StudentModel(nn.Module):             #needed for training

    #constructor method, for which are needed the word embedding vectors, the dimensions of the two linear
    #layers, the dropout probabilty p, a flag to choose if the lstm layer must be bidirectonal, the number
    #of layers of the lstm layer and the loss function (but these last 4 already have a default value)
    def __init__(self,embeddings,   #word embedding vectors
                 pos_embeddings=None,    #pos embedding vectors
                 hidden1=512,           #dimension of the first hidden layer
                 hidden2=512,           #dimension of the second hidden layer
                 p=0.0,             #probability of dropour layer
                 bidirectional=False,   #flag to decide if the LSTM must be bidirectional
                 lstm_layers=1,         #layers of the LSTM
                 loss_fn=torch.nn.CrossEntropyLoss(ignore_index=13)):   #loss function
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.pos_embeddings = None if pos_embeddings == None else nn.Embedding.from_pretrained(pos_embeddings,freeze=False)     #choose if creating or not an embedding layer
        hidden1 = hidden1*2 if bidirectional else hidden1                                                                       #based on wether pos embeddings were created
        input_dim = embeddings.size(1) if not self.pos_embeddings else embeddings.size(1)+pos_embeddings.size(1)                #or not
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden1, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.lin1 = nn.Linear(hidden1, 13)
        #self.lin2 = nn.Linear(hidden2,13)
        self.loss_fn = loss_fn
        self.dropout = nn.Dropout(p=p)

    #forward method, automatically called when calling the instance
    #it takes the Xs and their length in batches
    def forward(self,X,X_len):
        #lemmas,pos = X[...,0],X[...,1]                                  #separating pos from lemmas
        embedding = self.embedding(X)                                #expanding the words from indices to embedding vectors
        '''
        if self.pos_embeddings != None:                                     #in the case I'm using pos embeddings, I pass their indexes through their own embedding layer
            pos_embeddings_out1 = self.pos_embeddings(pos)                 #and then concatenate them to the corresponding words
            
            embedding1 = torch.cat([embedding,pos_embeddings_out1],dim=-1)
        '''
        lstm_out = self.lstm(embedding)[0]
        batch_size, sentence_len, hidden_size = lstm_out.shape  #sentence length here is taken to remove padding
        flattened1 = lstm_out.reshape(-1, hidden_size)        #the output of the lstm is flattened (batch size,sentence length,lstm output dimension)->
                        #                                       (batch size,sentence length*lstm output dimension)
        last_word_relative_indices = X_len - 1                #in order to have the last index of each sentence in the batch (excluding padding), we subtract one to the length
        sentences_offsets = torch.arange(batch_size, device=device) * sentence_len   #we compute the offsets, or absolute indexes, where each sentence starts
        vec_sum_index = sentences_offsets + last_word_relative_indices     #we take the last token of each sentence, which is the hidden
                                                                            #state of the LSTM which summarize the content of the sentence
        #vec_sum = flattened1[vec_sum_index]                         #we use the index to retrieve the vector which summarizes the sentence
        ##out = self.dropout(torch.cat([vec_sum1,vec_sum2],dim=-1))
        out = self.dropout(flattened1)
        out = torch.relu(out)
        #print(out.size())
        out = self.lin1(out)

        #out = self.lin2(out)
        out = out.squeeze(1)
        #print(out.size())
        out = torch.softmax(out,dim=-1)
        return out
    '''
    
    
    #predict function
    #takes the sentence pairs in their original form (each pair is actually a dictonary with several entries)
    #and outputs a list of all the predictions
    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!
        glove = create_glove()                          #creation of the glove vocabulary
        embeddings, word2idx = create_embeddings(glove) #creation of word embeddings and corresponding dictionary word:index
        pos_embeddings, pos2idx = None, None            #placeholder for pos embeddings and corresponding dictionary pos:index
        #if w_pos: pos_embeddings, pos2idx = create_pos_embeddings() #if the pos is to be used, create pos_embeddings and pos2idx
        dataloader = StudentDataset(sentence_pairs,word2idx,pos2idx,test=True).dataloader(batch_size=64)   #instantiating the dataset with the sentence pairs and loading it with the custom dataloader
        preds = list()  #list to append all the predictions
        with torch.no_grad():
            for batch in dataloader:
                batch_x1 = batch[0]     #separating first and second sentences
                batch_x2 = batch[1]
                batch_xlen1 = batch[2]  #separating lengths of first and second sentences
                batch_xlen2 = batch[3]
                batch_y = batch[4]
                pred = self(batch_x1, batch_x2, batch_xlen1, batch_xlen2)
                preds += torch.round(pred).tolist()     #adding the predictions to preds after rounding in order to have 0 or 1
        preds = [str(bool(pred)) for pred in preds]     #flattening preds (which was a list of batches of predictions)
        return preds
    '''
#trainer class
class Trainer():

    def __init__(self,model,optimizer):
        self.model = model
        self.optimizer = optimizer

    #train function, takes two dataloaders for trainset and devset in order to train on the trainset and
    #check at every epoch how the training is affecting performance on the dev set, to avoid overfitting
    #I use the patience mechanism to stop after 5 times the accuracy on devset goes down, since I noticed
    #that after that point it just gets worse
    def train(self,train_loader, dev_loader,patience, epochs=10):
        loss_history = [[], []]             #lists to save trainset and devset loss in order to plot the graph later
        accuracy_history = [[], []]         #lists to save trainset and devset accuracy in order to plot the graph later
        patience_counter = 0                #counter to implement patience
        for i in range(epochs):
            losses = []                     #list to save the loss for each batch
            hit = 0                         #counter for correct predictions
            total = 0                       #counter for all predictions
            for batch in train_loader:
                batch_x = batch[0]         #separating first from second sentences
                batch_xlen = batch[1]      #separating lengths of first and second sentences
                labels = batch[2]          #taking the ground truth
                ids = batch[3]
                self.optimizer.zero_grad()  #setting the gradients to zero for each batch
                logits = self.model(batch_x, batch_xlen) #predict
                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)
                #print(logits.size(),labels.size())
                #print(logits)
                #print(labels)
                loss = self.model.loss_fn(logits, labels)                #calculating the loss
                for j in range(len(logits)):                                      #checking the number of hits in order to compute accuracy
                    total += 1
                    if torch.argmax(logits[j]) == labels[j]: hit += 1
                loss.backward()             #backpropagating the loss
                self.optimizer.step()       #adjusting the model parameters to the loss
                losses.append(loss.item())  #appending the losses to losses
            accuracy = hit/total            #computing accuracy
            accuracy_history[0].append(accuracy)    #appending accuracy to accuracy history
            mean_loss = sum(losses) / len(losses)   #computing the mean loss for each epoch
            loss_history[0].append(mean_loss)       #appending the mean loss of each epoch to loss history
            metrics = {'mean_loss': mean_loss, 'accuracy': accuracy}    #displaying results of the epoch
            print(f'Epoch {i}   values on training set are {metrics}')
            #the same exact process is repeated on the instances of the devset, minus gradient backpropagation and optimization of course
            hit = 0
            total = 0
            with torch.no_grad():
                #RESET LOSSES????
                for batch in dev_loader:
                    batch_x = batch[0]
                    batch_xlen = batch[1]
                    labels = batch[2]
                    ids = batch[3]
                    logits = self.model(batch_x, batch_xlen)
                    logits = logits.view(-1, logits.shape[-1])
                    labels = labels.view(-1)
                    max_logits = torch.argmax(logits,dim=1)
                    loss = self.model.loss_fn(logits, labels)
                    losses.append(loss.item())
                    for j in range(len(logits)):
                        total += 1
                        if torch.argmax(logits[j]) == labels[j]: hit += 1
            mean_loss = sum(losses) / len(losses)
            loss_history[1].append(mean_loss)
            accuracy = hit / total
            accuracy_history[1].append(accuracy)
            metrics = {'mean_loss': mean_loss, 'accuracy': accuracy}
            print(f'            final values on the dev set are {metrics}')
            if len(accuracy_history[1]) > 1 and accuracy_history[1][-1] < accuracy_history[1][-2]:
                patience_counter += 1
                if patience == patience_counter:
                    print('-----------------------------EARLY STOP--------------------------------------------')
                    break
                else:
                    print('------------------------------PATIENCE---------------------------------------------')

        return {
            'loss_history': loss_history,
            'accuracy_history': accuracy_history
        }





#utility function to plot accuracy and loss
def plot_logs(history,param):
    plt.figure(figsize=(8, 6))
    train_param = history[0]
    test_param = history[1]
    plt.plot(list(range(len(train_param))), train_param, label='Train '+param)
    plt.plot(list(range(len(test_param))), test_param, label='Test '+param)
    plt.title('Train vs Test '+param)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.show()
model_path = "./myModel.ckpt"
glove = create_glove()                                          #create glove dictionary
embeddings,word2idx = create_embeddings(glove)                  #create and indexing embeddings
print(len(embeddings))
print("CREATED VOCABULARY")
model = StudentModel(embeddings,p=0.5).to(device)         #instantiating the model
print("CREATED MODEL")
#'''


train_dataset = SentenceDataset(sentences_path=TRAIN_PATH,vectors=embeddings,word2idx=word2idx) #instantiating the training dataset

print("CREATED TRAIN DATASET")

dev_dataset = SentenceDataset(sentences_path=DEV_PATH,vectors=embeddings,word2idx=word2idx,test=True)       #instantiating the dev dataset
print("CREATED DEV DATASET")
train_dl = train_dataset.dataloader(batch_size=64)                         #instantiating the dataloaders
dev_dl = dev_dataset.dataloader(batch_size=64)

print("CREATED DATALOADERS")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00) #instantiating the optimizer
trainer = Trainer(model,optimizer)                                              #instantiating the trainer
histories = trainer.train(train_loader=train_dl,dev_loader=dev_dl,patience=10,epochs=100)    #training
params = ['loss', 'accuracy']                                                   #plotting the metrics
for param in params:
    plot_logs(histories[param + '_history'], param)
torch.save(model.state_dict(), model_path)                                      #saving the model



'''

model.load_state_dict(torch.load(model_path,map_location=device))

#'''



