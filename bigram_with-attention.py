
import os
import math
from tqdm import tqdm

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
block_size = 32 # context
n_embed = 192 # embedding size (n_embed//n_heads = head_size)
n_heads = 6 # self-attention heads
n_blocks = 4 # transformer blocks (layers)
max_new_tokens = 1000
max_iters = 3000
eval_interval = 500
eval_iters = 200
lr = 1e-3
dropout_rate = 0.1

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
    model.eval() # eval mode

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



# one self-attention head
class AttentionHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,head_size) - unit gaussian (var:1)
        q = self.query(x) # (B,T,head_size) - unit gaussian (var:1)

        # data dependent weighted aggregation (communication)
        # dot product between all queries and all keys
        # ! need to transpose k (last 2 dimensions: dim -1, dim -2)
        # weight = q @ k.transpose(-2, -1)
        # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)

        # scaled attention
        weight = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_size))
        # with naive matmul, weight is not ug (var:head_size)
        # -> maintain a consistent level of variance in the dot product distribution
        # avoid excessively large activations.
        # softmax sharpens large values - we would converge to just one very high value and aggregate info only from that respective node. 
        # instead, we want to aggregate info from multiple nodes that are important for the prediction to different extents.

        # mask future tokens (decoder block)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)

        # softmax
        # important tokens will get higher values
        # more info from those tokens will be aggregated into current timestep idx
        weight = F.softmax(weight, dim=-1) # (B,T,T)

        weight = self.dropout(weight) # prevent some of the node from communicating
        
        # weighted average of values
        # propagate through lin layer and aggregate the resulting value, not the token itself
        v = self.value(x) # (B,T,head_size)
        out = weight @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)

        return out
            


# multi-head attention
class MultiHeadAttention(nn.Module):
    ''' multiple attention heads running in parallel '''
    
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embed, n_embed) # for skip connections
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out) # project back to the residual pathway
        out = self.dropout(out)
        return out



# feedforward: multi-layer perceptron
class FeedForward(nn.Module):
    ''' linear layer followed by non-linearity (ReLU) '''
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4), # dim (channel size) of the inner layer is 4x (from Transformer paper)
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed), # projection back to the residual pathway
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # compute at the per token level, after aggregation of info via attention
        return self.net(x)
        


# block: alternate communication + computation
class TransformerBlock(nn.Module):
    ''' aggregation via attention (communication) followed by feedforward (computation) '''

    def __init__(self, n_embed, n_heads):
        super().__init__()
        
        # head_size*n_heads = n_embed (concatenated in multi-head attention)
        head_size = n_embed // n_heads
        self.selfattention_heads = MultiHeadAttention(n_heads, head_size)
        self.feedforward = FeedForward(n_embed)
        self.layernorm1 = nn.LayerNorm(n_embed)
        self.layernorm2 = nn.LayerNorm(n_embed)


    def forward(self, x):
        #x = self.selfattention_heads(x)
        #x = self.feedforward(x)

        ''' residual connections '''
        x = x + self.selfattention_heads(x)
        x = x + self.feedforward(x)

        ''' residual connections + layer normalization '''
        # variation from the Transformer paper:
        # layer normalization is applied directly on the input (instead of after attention/FF)
        x = x + self.selfattention_heads(self.layernorm1(x))
        x = x + self.feedforward(self.layernorm2(x))

        return x



# model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token (idx) reads off the logits for the next token (lookup table)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # each pos has its own embedding
        
        ''' one attention head '''
        #self.selfattention_head = AttentionHead(n_embed) one head
        ''' multiple attention heads '''
        #self.selfattention_heads = MultiHeadAttention(n_heads, n_embed//n_heads) # multiple heads/channels
        # n_heads of size n_embed//n_heads dimensional self-attention (which will be concatenated, resulting in n_embed)
        #self.feedforward = FeedForward(n_embed)
        ''' transformer blocks '''
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embed, n_heads) for _ in range(n_blocks)]
        )
        # ! deep network -> optmization issues (vanishing gradients)
        # -> add residual connections (skip connections) + layer norm

        self.final_layernorm = nn.LayerNorm(n_embed) # norm always at the end of the network

        # linear layer that decoded into vocabulary space (get logits for each token in the vocab)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_embed = self.token_embedding_table(idx) # (B,T,C:n_embed)
        pos_embed = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C:n_embed)
        x = tok_embed + pos_embed # (B,T,C:n_embed)
        
        ''' one attention head '''
        #x = self.selfattention_head(x) # one head (B,T,C:n_embed)
        ''' multiple attention heads '''
        #x = self.selfattention_heads(x) # multiple heads (B,T,C:n_embed)
        #x = self.feedforward(x) # (B,T,C:n_embed)
        ''' transformer blocks '''
        x = self.blocks(x) # (B,T,C:n_embed)

        logits = self.lm_head(x) # (B,T,C:vocab_size)
        
        # Batch: batch_size
        # Time: block_size
        # Channels: vocab_size
        # logits are the scores to determine the next character in the sequence
        # based on identity of a single token (no context at this point)
        # we can predict what token comes next (to some extent)
        # because certain tokens are known to follow other more frequently

        if targets is None:
            # will just return the logits
            loss = None
        else:
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
            # constrain idx to the last block_size tokens
            # otherwise, the pos embedding table will run out of indices
            # -> never pass more than block_size elements to the model
            idx_con = idx[:, -block_size:]

            # get the predictions
            # (loss will be ignored; no correct targets to compare with)
            logits, loss = self(idx_con)

            # bigram model: focus only on the last time step (last idx == last token)
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution num_samples
            # for each batch dim, there will be 1 prediction (next token)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
model = model.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training loop
for iter in tqdm(range(max_iters)):
    
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')

# generate text
context = torch.zeros((1,1), dtype=torch.long, device=device)

print(tokenizer.decode(model.generate(context, max_new_tokens)[0]))
