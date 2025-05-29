import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=1000):
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
    def __init__(self, input_features, output_features, output_window,  dropout=0.2, num_layers=3, hidden_dim=128, num_heads=4, teacher_forcing_ratio=1.0, target_coin_index=0):
        super().__init__()
        self.output_features = output_features
        self.output_window = output_window
        self.target_coin_index = target_coin_index
        self.hidden_dim = hidden_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # project input and outputs to required model dimensions
        # and project model output to required data dimensions
        self.input_proj = nn.Linear(in_features=input_features, out_features=hidden_dim)
        self.output_proj = nn.Linear(in_features=output_features, out_features=hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_features)

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        self.pos_decoder = PositionalEncoding(hidden_dim, dropout=dropout)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=512, dropout=dropout, batch_first=True)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward_with_target(self, src, tgt):
        src = src.permute(0, 2, 1)
        src = self.input_proj(src) * math.sqrt(self.hidden_dim)
        src = self.pos_encoder(src)
    
        tgt = tgt.permute(0, 2, 1)
        tgt = self.output_proj(tgt) * math.sqrt(self.hidden_dim)
        tgt = self.pos_decoder(tgt)
        
        tgt_mask = self.generate_square_subsequent_mask(self.output_window).to(src.device)
        
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.output_linear(out)
        
        return out.permute(0, 2, 1)

    def forward_without_target(self, src):
        batch_size = src.size(0)
        src = src.permute(0, 2, 1)
        
        last_input = src[:, -1, :][:, self.target_coin_index*self.output_features:(self.target_coin_index+1)*self.output_features]
        
        src = self.input_proj(src) * math.sqrt(self.hidden_dim)
        src = self.pos_encoder(src)

        output = torch.zeros(batch_size, self.output_window + 1, self.output_linear.out_features, device=src.device)

        # Initialize index 0 with last encoder input
        output[:, 0, :] = last_input

        full_tgt_mask = self.generate_square_subsequent_mask(self.output_window + 1).to(src.device)

        for t in range(self.output_window):
            decoder_input = output[:, :t+1+1, :]
            decoder_input = self.output_proj(decoder_input) * math.sqrt(self.hidden_dim)
            decoder_input = self.pos_decoder(decoder_input)

            tgt_mask = full_tgt_mask[:t+1+1, :t+1+1]

            out = self.transformer(src, decoder_input, tgt_mask=tgt_mask)
            out = self.output_linear(out)

            output[:, t+1, :] = out[:, -1, :]  # write to index t+1

        # Discard index 0 (the seed), return predictions only
        return output[:, 1:, :].permute(0, 2, 1)

    def set_teacher_forcing_ratio(self, new_value):
        self.teacher_forcing_ratio = new_value

    def call(self, x, y):
        if y is not None and torch.rand(1) < self.teacher_forcing_ratio:
            output = self.forward_with_target(x, y)
        else:
            output = self.forward_without_target(x)
        
        return output