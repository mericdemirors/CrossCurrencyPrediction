import torch
import torch.nn as nn

class CoinWiseCrossAttentionLSTM(nn.Module):
    def __init__(self, output_features, input_window, output_window, dropout=0.2, num_layers=3, hidden_dim=128, num_heads=4, target_coin_index=0, num_coins=4):
        super().__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.output_features = output_features
        self.num_coins = num_coins
        self.target_coin_index = target_coin_index

        # one LSTM per coin
        self.lstm_blocks = nn.ModuleList([
            nn.LSTM(input_size=self.output_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
            ) for _ in range(self.num_coins)
        ])

        # attention to merge coin pipelines
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_features * output_window)

    def forward(self, x):
        batch_size = x.size(0)

        # one input per coin pipeline
        x = x.view(batch_size, self.num_coins, self.output_features, self.input_window)
        lstm_outputs = []

        # pass all coin pipelines and merge their last outputs in one tensor
        for i in range(self.num_coins):
            coin_input = x[:, i]
            coin_input = coin_input.permute(0, 2, 1)
            _, (h_n, _) = self.lstm_blocks[i](coin_input)
            lstm_outputs.append(h_n[-1])

        lstm_stack = torch.stack(lstm_outputs, dim=1)

        # apply attention to merged coin pipeline last features
        attn_out, _ = self.attention(lstm_stack, lstm_stack, lstm_stack)

        # get the applied attention to the target coin pipeline output
        target_coin = attn_out[:, self.target_coin_index, :]
        
        output = self.fc(target_coin)
        
        output = output.view(batch_size, self.output_features, self.output_window)

        return output
    
    def call(self, x, y):
        return self(x)