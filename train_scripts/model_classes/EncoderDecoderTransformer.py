import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, input_features, output_features, output_window,  dropout, num_layers, hidden_dim, num_heads, teacher_forcing_ratio, output_col_indices_in_input_cols):
        super().__init__()
        self.output_features = output_features
        self.output_window = output_window
        self.hidden_dim = hidden_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.output_col_indices_in_input_cols = output_col_indices_in_input_cols

        # project input and outputs to required model dimensions
        # and project model output to required data dimensions
        self.input_proj = nn.Linear(in_features=input_features, out_features=hidden_dim)
        self.output_proj = nn.Linear(in_features=output_features, out_features=hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_features)

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        self.pos_decoder = PositionalEncoding(hidden_dim, dropout=dropout)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=hidden_dim*output_features, dropout=dropout, batch_first=True)

    def forward_with_target(self, src, tgt):
        src = src.permute(0, 2, 1)
        tgt = tgt.permute(0, 2, 1)

        # shifting the target to one index right, and then putting the last input data at the first index
        tgt = tgt.roll(1, 1)
        # replace the first time data at every batch, with the first self.output_features values of the last time stamp of every batch
        tgt[:,0] = src[:,-1,self.output_col_indices_in_input_cols]

        src = self.input_proj(src) * math.sqrt(self.hidden_dim)
        src = self.pos_encoder(src)

        tgt = self.output_proj(tgt) * math.sqrt(self.hidden_dim)
        tgt = self.pos_decoder(tgt)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.output_window).to(src.device)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_is_causal=True)
        out = self.output_linear(out)
        
        return out.permute(0, 2, 1)

    def forward_without_target(self, src):
        batch_size = src.size(0)
        src = src.permute(0, 2, 1)

        last_input = src[:, -1, self.output_col_indices_in_input_cols]
        
        src = self.input_proj(src) * math.sqrt(self.hidden_dim)
        src = self.pos_encoder(src)

        output = torch.zeros(batch_size, self.output_window + 1, self.output_linear.out_features, device=src.device)

        # Initialize index 0 with last encoder input
        output[:, 0, :] = last_input

        full_tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.output_window + 1).to(src.device)

        for t in range(self.output_window):
            decoder_input = output[:, :t+1, :].clone()
            
            decoder_input = self.output_proj(decoder_input) * math.sqrt(self.hidden_dim)
            decoder_input = self.pos_decoder(decoder_input)

            tgt_mask = full_tgt_mask[:t+1+1, :t+1+1]

            out = self.transformer(src, decoder_input, tgt_mask=tgt_mask, tgt_is_causal=True)
            out = self.output_linear(out)

            output[:, t+1, :] = out[:, -1, :]  # write to index t+1

        # Discard index 0 (the seed), return predictions only
        out = output[:, 1:, :].permute(0, 2, 1)
        return out

    def set_teacher_forcing_ratio(self, new_value):
        self.teacher_forcing_ratio = new_value

    def call(self, x, y):
        if torch.rand(1) < self.teacher_forcing_ratio:
            output = self.forward_with_target(x, y)
        else:
            output = self.forward_without_target(x)
        
        return output