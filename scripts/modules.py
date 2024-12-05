import os,sys,time
import numpy as np
import torch
import torch.nn as nn

# Attention module
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads  
        
        assert (self.head_dim * num_heads == embedding_dim), \
                "Embedding dimension must be divisible by number of heads"

        ## Initialize the weights for the values, keys, and queries
        self.Weights_values = nn.Linear(in_features=self.head_dim, out_features=self.head_dim, bias=False)
        self.Weights_keys = nn.Linear(in_features=self.head_dim, out_features=self.head_dim, bias=False)
        self.Weights_query = nn.Linear(in_features=self.head_dim, out_features=self.head_dim, bias=False)

        ##  Fully connected layer to project input patches to the hidden size dimension
        self.fc_output = nn.Linear(in_features=self.head_dim, out_features=self.head_dim, bias=False)    

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys) 
        queries = self.queries(query) 

        ## Split the embedding into self.num_heads different pieces
        ## shape: (N, v/q/k_len, heads, head_dim)
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        ## Compute attention scores
        ## the equation is (q * k^T)/sqrt(d_model) * v

        # einsum and matmul are similar function for matrix multiplication
        Q_KT = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            Q_KT = Q_KT.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(Q_KT / (self.embedding_dim ** (1 / 2)), dim=3)
        
        # Reshape the attention score: (N, heads, query_len, key_len)
        attension_score = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )

        attension_score = self.fc_out(attension_score)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return attension_score
    
# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, 
                 embedding_dim, num_heads, 
                 forward_expansion, dropout,
                 device, 
                 ):
        super().__init__()

        ##self.attention = SelfAttention(embedding_dim, num_heads)
        self.multihead_attn = torch.nn.MultiheadAttention(
                        embed_dim=embedding_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        device=device)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_dim, embedding_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_query, input_value, input_key, \
                mask=None):
        # attention_out = self.attention(input_query, input_key, input_value, mask)
        attenion_out, _ = self.multihead_attn(query = input_query, \
                                            value = input_value,
                                            key = input_key, 
                                            key_padding_mask=mask)
        x_in = self.norm1(input_query + attenion_out)
        x_out = self.feed_forward(x_in)
        x_result = self.norm2(x_out+x_in)
        return x_result

# Positional encoding
class PositionalEncoding_old(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()

        position_endoding = torch.zeros(max_len, dim_model)
        position_list = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=dim_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index / dim_model) 
        position_endoding[:, 0::2] = torch.sin(position_list * div_term)
        position_endoding[:, 1::2] = torch.cos(position_list * div_term)

        self.register_buffer('position_endoding', position_endoding)

    def forward(self, x):
        return x + self.position_endoding[:x.size(0), :]
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()

        position_endoding = torch.zeros(max_len, dim_model)
        position_list = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=dim_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index / dim_model) 
        position_endoding[:, 0::2] = torch.sin(position_list * div_term)
        position_endoding[:, 1::2] = torch.cos(position_list * div_term)

        self.register_buffer('position_endoding', position_endoding)

    def forward(self, x):
        return x + self.position_endoding[:x.size(1),  :]
    
class Reduce_dim(nn.Module):
    # Reduce the dimension for the spectra
    # from autoencoder
    def __init__(self, input_size, latent_dim):
        super(Reduce_dim, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class Expand_dim(nn.Module):
    # Expand the latent space spectrum to the reconstructed spectrum
    def __init__(self, input_size, latent_dim):
        super(Expand_dim, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

class Encoder(nn.Module):
    def __init__(self, 
                 embedding_dim, num_heads, num_layers, 
                 forward_expansion, dropout, device):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.layers = nn.ModuleList([
                    TransformerBlock(embedding_dim, num_heads, 
                                    forward_expansion, dropout, device
                                    ) for _ in range(num_layers)  ])

    def forward(self, x, mask=None):
        x = self.norm1(x)

        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x

class Decoder(nn.Module):
    def __init__(self, 
                 embedding_dim, num_heads, num_layers, 
                 forward_expansion, dropout, device):
        super().__init__()

        #self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.layers = nn.ModuleList([
                    TransformerBlock(embedding_dim, num_heads, 
                                    forward_expansion, dropout, device
                                    ) for _ in range(num_layers)  ])

    def forward(self, x, encoder_val, encoder_key, mask=None):
        for layer in self.layers:
            x = layer(x, encoder_val, encoder_key, mask)
        return x

class SpecTransformer_old(nn.Module):
    def __init__(self, 
                 input_size,
                 label_size,
                 embedding_dim = 128, 
                 num_heads = 8,
                 num_layers = 3, 
                 forward_expansion = 4, 
                 dropout = 0, 
                 device='cpu',
                 ):
        super().__init__()

        max_length = input_size + label_size
        self.input_size = input_size
        self.label_size = label_size

        self.device = device
        self.encoder = Encoder(embedding_dim, num_heads, num_layers, 
                               forward_expansion, dropout, device)
        self.decoder = Decoder(embedding_dim, num_heads, num_layers, 
                               forward_expansion, dropout, device)
        
        latent_dim = embedding_dim-label_size
        self.reduce_dim = Reduce_dim(input_size, latent_dim)
        self.expand_dim = Expand_dim(input_size, latent_dim)

        self.positional_encoding = PositionalEncoding_old(embedding_dim, max_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, params):
        # x: (nsamp, input_size) or (1, nsamp, input_size)
        # params: (nsamp, label_size) or (1, nsamp, label_size)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(params.shape) == 2:
            params = params.unsqueeze(0)
        #elif len(params.shape) == 1:

        # Check the x (1, nsamp, input_size) and params (1, nsamp, label_size)
        #assert x.shape[0] == 1, "Batch size must be 1"      
        #assert params.shape[0] == 1, "Batch size must be 1"
        assert x.shape[2] == self.input_size, "Input size does not match"
        assert params.shape[2] == self.label_size, "Label size does not match"
            
        x_latent = self.reduce_dim(x)
        x_add = torch.cat((x_latent, params), dim=2)
        x_positioned = self.positional_encoding(x_add)
        x_in = self.dropout(x_positioned)

        encoder_seq = self.encoder(x_in)
        x_out = self.decoder(x_in, encoder_seq, encoder_seq)

        params_out = x_out[:, :,  -self.label_size:]
        x_toexpand = x_out[:, :, :-self.label_size]
        x_spec = self.expand_dim(x_toexpand)
        
        return x_spec, params_out

class SpecTransformer(nn.Module):
    def __init__(self, 
                 input_size,
                 label_size,
                 latent_dim=64,
                 embedding_dim = 128,
                 max_length = 1000, 
                 num_heads = 8,
                 num_layers = 3, 
                 forward_expansion = 4, 
                 dropout = 0, 
                 device='mps',
                 ):
        super().__init__()

        self.input_size = input_size
        self.label_size = label_size

        self.device = device
        self.encoder = Encoder(embedding_dim, num_heads, num_layers, 
                               forward_expansion, dropout, device)
        self.decoder = Decoder(embedding_dim, num_heads, num_layers, 
                               forward_expansion, dropout, device)
        
        self.reduce_dim = Reduce_dim(input_size, latent_dim)
        self.expand_dim = Expand_dim(input_size, latent_dim)

        self.embedding = nn.Linear(1, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, 1)

        self.positional_encoding = PositionalEncoding(embedding_dim, max_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, params):
        # Input shape
        # x: (nsamp, npix)
        # params: (nsamp, nparams)

        # Reduce the dimension of the spectrum
        # x_latent: (nsamp, latent_dim)
        x_latent = self.reduce_dim(x)
        
        # Add the parameters/labels to the reduced spectrum
        # x_add: (nsamp, latent_dim + nparams, 1)
        x_add = torch.cat((x_latent, params), dim=1)
        x_add = x_add.unsqueeze(-1)
        
        # Embedding and positional encoding
        # x_embedded: (nsamp, latent_dim + nparams, embedding_dim)
        # x_positioned: (nsamp, latent_dim + nparams, embedding_dim)
        x_embedded = self.embedding(x_add)
        x_positioned = self.positional_encoding(x_embedded)
        
        x_in = self.dropout(x_positioned)

        # Transformer encoder-decoder
        encoder_seq = self.encoder(x_in)
        x_out = self.decoder(x_in, encoder_seq, encoder_seq)

        # Take the labels out
        # params_out: (nsamp, nparams, embedding_dim)
        params_out = self.fc_out(x_out[:,  -self.label_size:, :])

        # Expand the dimension of the spectrum
        # x_spec: (nsamp, npix)
        x_toexpand = self.fc_out(x_out[:, :-self.label_size, :])
        x_spec = self.expand_dim(x_toexpand.squeeze(-1))

        return x_spec, params_out

def masked_mse_loss(y_true, y_pred, mask):
    mse = ((y_true - y_pred) ** 2) * mask
    return mse.sum() / mask.sum()

def masked_gaussian_likelihood(y_true, y_pred, mask, sigma):
    mask_tmp = ~(mask == 0)
    loss = 0.5 * (torch.log(sigma[mask_tmp]**2) + (y_true - y_pred)[mask_tmp]**2 / (sigma[mask_tmp]**2))
    return loss.nanmean()


def generate_missing_pixel(Xarr, missing_rate=0.2):
    nsamp = Xarr.shape[0]
    X_missed = torch.clone(Xarr)
    for obj in range(nsamp):
        available_val_ind = np.where(Xarr[obj] != 0)[0]
        missing_npix = int(missing_rate*len(available_val_ind))
        missing_ind_start = np.random.choice(len(available_val_ind)-missing_npix)+available_val_ind[0]
        X_missed[obj, missing_ind_start:missing_ind_start+missing_npix] = 0
    return X_missed