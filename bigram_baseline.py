
import os
import torch
import torch.nn as nn
from torch.nn import functional as F #softmax

from data import DataHandler
from tokenizer import CharacterTokenizer



dev = False
torch.manual_seed(42)

# setup (data, tokenizer)
data_path = os.path.join(os.getcwd(), 'data')
data_handler = DataHandler(data_path)
data_handler.load_data()
data = data_handler.data
if dev:
    data = data[:1000]

tokenizer = CharacterTokenizer(data)
vocab_size = tokenizer.vocab_size

# hyperparameters
batch_size = 16
block_size = 8
max_iters = 2000
eval_interval = 200
eval_iters = 100
lr = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if dev:
    max_iters = 50
    eval_interval = 10
    eval_iters = 10

# encode data + split
data = tokenizer.encode(data)
train, val = data_handler.get_splits(data, train_size=0.9)

x = train[:block_size]
y = train[1:block_size+1] # shift by 1 to include next character (target)


# helper functions
def get_batch(split):
    '''
    generate a batch of data of inputs x and targets y

    select random chunks of the train/val set
    (batch_size number of chunks)

    random set of offsets to start looking at text from
    ix (with batch size 4): tensor([ 73834, 121141,  90465, 125761])
    '''
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # +1 because y follow last x of the context
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # do not do backpropagation
def estimate_loss():
    ''' 
    average loss over multiple batches 
    -> more robust measurement of the current loss (depends on the batch)
    '''
    loss_tracker = {}
    model.eval() # eval mode (currently makes no difference - only lookup table)

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y) # forward pass
            losses[i] = loss.item()
        loss_tracker[split] = losses.mean()

    model.train()
    return loss_tracker


# model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token (idx) reads off the logits for the next token (lookup table)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        # Batch: batch_size 4
        # Time: block_size 8
        # Channels: vocab_size 84
        # logits are the scores to determine the next character in the sequence
        # based on identity of a single token (no context at this point)
        # we can predict what token comes next (to some extent)
        # because certain tokens are known to follow other more frequently

        if targets is None:
            # will just return the logits
            loss = None
        else:
            # logits: the dimension corresponding to the correct target
            # should have a high number (the others low)
            # ! need to resize to match input that pytorch expects for CE
            # ! for multidim input, the channel should be the 2nd dim (B, C, T)
            # ! instead, transform into a 2d array

            # unpack
            B, T, C = logits.shape
            # 3D to 2D
            logits = logits.view(B*T, C)
            # 2D (B, T) to 1D
            targets = targets.view(B*T)
            # negative log likelyhood
            loss = F.cross_entropy(logits, targets) # measure quality of logits wrt targets
            # knowing the vocab size, we can estimate the loss
            # -ln(1/vocab_size)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        '''
        idx: (B, T) array of indices in the current context
        generate: extend array to be B, T+1 (+2,+3, ...)
            continue generation in the time dimension
            for each batch dimension
        '''
        for _ in range(max_new_tokens):
            # get the predictions
            # (loss will be ignored; no correct targets to compare with)
            logits, loss = self(idx)

            # focus only on the last time step (last idx == last token)
            # (because it is a bigram model)
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution num_samples
            # for each batch dim, there will be 1 prediction (next token)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
model = model.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training loop
for iter in range(max_iters):
    
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        losses = estimate_loss() # finish function
        print(f'Iter {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')

# generate text
context = torch.zeros((1,1), dtype=torch.long, device=device)

print(tokenizer.decode(model.generate(context, 500)[0]))