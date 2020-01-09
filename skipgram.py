import torch
import jieba
import random
import numpy as np
from collections import Counter

stopwords = ['的', '阿']

chinese_vocab_size = 10000

embed_size = 300

def readdata(sfile):
    words = []
    data = open(sfile, 'r', encoding='utf-8')
    for line in data:
        token = jieba.cut(line)
        if token in stopwords:
            continue
        words.extend(token)
    return words

def build_vocab(vfile):
    vocab = {}
    vocabf = open(vfile, 'r', encoding='utf-8')
    for line in vocabf:
        word, id_ = line.split(' ')
        vocab[word] = id_
    return vocab


def word_to_id(words):
    ids = []
    for word in words:
        if word not in vocab:
            id_ = vocab['unk']
        else:
            id_ = vocab[word]
        ids.append(id_)
    return ids

def get_label(batch, id_, window_size):
    # r = w
    r = np.random.randint(1, window_size*1)
    start_index = id_ -r if (id_ - r)>0 else 0
    end_index = id_ +r if (id_ + r)>len(batch) else batch[-1]
    labels = batch[start_index:id_] + batch[id_+1:end_index]
    return labels


def get_batch(tokens, batch_size, window_size=3):
    n_batches = int(len(tokens) / batch_size)
    tokens = tokens[0:n_batches*batch_size]

    for index in range(0, len(tokens), batch_size):
        x, y = [], []
        batch = tokens[index, index*batch_size]
        for id_ in range(len(batch)):
            batch_x = batch(id_)
            batch_y = get_label(batch, id_, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
            assert len(x) == len(y)
        yield x, y


class SkipGram(torch.nn.Module):

#1. Take one-hot to embeding with product matrix
#2. embed with word distribution to predict Skip-Gram

    def __init__(self, vocab_size, embed_size, noise_probs):
        super().__init__()
        self.noise_probs = noise_probs

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = torch.nn.Embedding(embed_size, chinese_vocab_size)
        self.in_embed.weight.data.uniform_(-1, 1)
        
        self.out_embed = torch.nn.Linear(vocab_size, embed_size)
        self.out_embed.weight.data.uniform_(-1, 1)
        #self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward_input(self, x):
        input_vector = self.in_embed(x)
        return input_vector
    
    def forward_output(self, out):
        output_vector = self.out_embed(out)
        return output_vector
    
    def forward_noise(self, batch_size, sample_num):
        #multionmial

        if self.noise_probs is None:
            noise_probs = torch.ones(self.vocab_size)
        else:
            noise_probs = self.noise_probs
        
        noise_words = torch.multinomial(noise_probs, batch_size*sample_num)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        noise_words = noise_words.to(device)
        noise_vector = self.out_embed(noise_words).view(batch_size, sample_num, self.embed_size)
        return noise_vector
    '''
    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        logout = self.log_softmax(scores)
        return logout
    '''
class NegativeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, out, noise):
        input_vector = x.view(batch_size, embed_size, 1)

        output_vector = out.view(batch_size, 1, embed_size)

        out_loss = torch.bmm(input_vector, output_vector).sigmoid().log()
        out_loss = out_loss.squeeze()

        noise_loss = torch.bmm(noise, x).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum()

        loss = -(out_loss*noise_loss).mean()

        return loss


data = readdata('data.txt')

vocab = build_vocab('vocab.txt')

tokens = word_to_id(data)

word_freqs = np.array(sorted(Counter(tokens).values(), reverse=True))

noise_probs = torch.from_numpy(word_freqs**0.75)/np.sum(word_freqs**0.75)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SkipGram(chinese_vocab_size, embed_size, noise_probs).to(device)

#criterion = torch.nn.NLLLoss()

criterion = NegativeLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

epoch = 10
batch_size = 32
sample_num = 5





for inputs, labels in get_batch(tokens, batch_size):
    inputs, labels = torch.LongTensor(inputs), torch.LongTensor(labels)
    inputs, labels = inputs.to(device), labels.to(device)

    #logout = model(inputs)
    input_vector = model.forward_input(inputs)
    output_vector = model.forward_output(labels)
    noise_vector = model.forward_noise(batch_size, sample_num)

    #loss = criterion(logout, labels)
    loss = criterion(input_vector, output_vector, noise_vector)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()