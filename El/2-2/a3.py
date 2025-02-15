import torch
import torch.nn as nn
import torch.nn.functional as F

# Example sentences
sentence_en = "I love AI ."
sentence_fr = "J' adore l'IA ."

# Word mappings (toy vocab)
word_map_en = {"<pad": 0, "I": 1, "love": 2, "AI": 3, ".": 4}
word_map_fr = {"<pad": 0, "J'": 1, "adore": 2, "l'IA": 3, ".": 4}

# Tokenization function
def tokenize(statement, word_map):
    tokens = [word_map[word] for word in statement.split()]
    print(f'Tokens for {statement}: {tokens}')
    return torch.tensor(tokens)

# Convert sentences to tensors
input_tensor = tokenize(sentence_en, word_map_en).unsqueeze(0)  # Shape: (1, sequence_length)
target_tensor = tokenize(sentence_fr, word_map_fr).unsqueeze(0)

# Define embedding dimension before using it
embedding_dim = 8  # Fix: Define embedding size

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        pos_enc = self.encoding[:, :x.size(1)]  # Slice for correct sequence length
        print("Positional Encoding:\n", pos_enc)
        output = x + pos_enc  # Add positional encoding to embeddings
        print("Final Output:\n", output)
        return output

# Create an embedding layer for input tokens
vocab_size = len(word_map_en)
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embedded_input = embedding_layer(input_tensor)  # Shape: (1, sequence_length, embedding_dim)

# Initialize positional encoding layer
pos_encoding_layer = PositionalEncoding(embedding_dim)  # Fix: `embedding_dim` is now defined
encoded_input = pos_encoding_layer(embedded_input)  # Apply positional encoding

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super( MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model//num_heads
        self.d_v = d_model//num_heads

        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model,d_model)
        self.value = nn.Linear(d_model,d_model)

        self.fc = nn.Linear(d_model,d_model)

    def forward(self,x,mask = None):
        batch_size,seq_len,_ = x.size()

        q = self.query(x).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        k = self.key(x).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        v = self.value(x).view(batch_size,seq_len,self.num_heads,self.d_v).transpose(1,2)

        attn_scores = torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(self.d_k,dtype = torch.float))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0,float('-inf'))
        attn_weights = F.softmax(attn_scores,dim = 1)

        attention_output = torch.matmul(attn_weights,v).transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        output = self.fc(attention_output)

        return output

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super(FeedForward,self).__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff = 512):
        super(EncoderLayer,self).__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.ff = FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x,mask=None):
        attn_output = self.mha(x,mask)
        x = self.norm1(x+attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x+ff_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=512):
        super(DecoderLayer, self).__init__()
        self.multihead1 = MultiHeadAttention(d_model, num_heads)  # Self-Attention
        self.multihead2 = MultiHeadAttention(d_model, num_heads)  # Encoder-Decoder Attention
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Step 1: Masked Self-Attention
        attn_output1 = self.multihead1(x, tgt_mask)
        x = self.norm1(x + attn_output1)

        # Step 2: Encoder-Decoder Attention
        attn_output2 = self.multihead2(x, encoder_output, src_mask)  # ✅ Use encoder_output
        x = self.norm2(x + attn_output2)

        # Step 3: Feed-Forward Network
        ff_output = self.ff(x)
        x = self.norm3(x + ff_output)

        return x

class Teransformer(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,num_encoder_layers,num_decoder_layers,max_len=5000):
        super(Teransformer,self).__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model,max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,num_heads) for _ in range(num_decoder_layers)])
        self.fc_out = nn.Linear(d_model,vocab_size)

    def forward(self,src,tgt,tgt_mask=None):
        src = self.pos_encoder(self.embedding(src))
        tgt = self.pos_encoder(self.embedding(tgt))

        encoder_output = src
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        decoder_output  = tcg
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output,encoder_output,tgt_mask = tgt_mask)

        output = self.fc_out(decoder_output)
        return output

import torch
import torch.nn.functional as F

def translate(input_sentence, word_map_en, word_map_fr, transformer):
    # 🔹 Step 1: Tokenize input sentence
    input_tensor = tokenize(input_sentence, word_map_en).unsqueeze(0)  # Shape (1, seq_len)

    # 🔹 Step 2: Generate a mask for the target sentence (lower triangular mask)
    tgt_mask = torch.tril(torch.ones((input_tensor.size(1), input_tensor.size(1)))).unsqueeze(0).unsqueeze(0)

    # 🔹 Step 3: Initialize a tensor for the target sentence (empty target sequence)
    target_tensor = torch.zeros((1, input_tensor.size(1)), dtype=torch.long)

    # 🔹 Step 4: Predict the output sentence (translation)
    output = transformer(input_tensor, target_tensor, tgt_mask)

    # 🔹 Step 5: Apply Softmax to get probabilities
    softmax_output = F.softmax(output, dim=-1)

    # 🔹 Step 6: Get predicted token indices (Argmax)
    predicted_tokens = torch.argmax(softmax_output, dim=-1)

    # 🔹 Step 7: Convert predicted tokens back to words
    reverse_word_map_fr = {v: k for k, v in word_map_fr.items()}  # Reverse word map
    translated_sentence = [reverse_word_map_fr[token.item()] for token in predicted_tokens[0] if token.item() != 0]

    return " ".join(translated_sentence)  # Join words into a sentence

# Define vocabulary sizes for English and French
vocab_size_en = len(word_map_en)  # Number of unique tokens in English vocabulary
vocab_size_fr = len(word_map_fr)  # Number of unique tokens in French vocabulary

# Define Transformer model parameters
d_model = 128  # Embedding dimension
num_heads = 8  # Number of attention heads
num_encoder_layers = 2  # Number of encoder layers
num_decoder_layers = 2  # Number of decoder layers

# Initialize the Transformer model
transformer = Transformer(vocab_size_en, d_model, num_heads, num_encoder_layers, num_decoder_layers)