import torch
import torch.nn as nn

class FeatureWiseCrossAttentionLSTM(nn.Module):
    def __init__(self, input_features, output_features, output_window, dropout=0.2, num_layers=3, hidden_dim=128, num_heads=4, target_coin_index=0):
        super().__init__()
        self.output_window = output_window
        self.output_features = output_features
        self.input_features = input_features
        self.num_groups = self.input_features // self.output_features
        self.target_coin_index = target_coin_index

        # one LSTM per coin feature
        self.feature_lstms = nn.ModuleList([
            nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
            for _ in range(self.input_features)
        ])

        # one attention to merge each coins' feature pipelines
        self.group_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
            for _ in range(self.num_groups)
        ])

        # attention to merge coin pipelines
        self.final_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_features * output_window)

    def forward(self, x):
        batch_size = x.size(0)
        feature_representations = []


        # pass all feature pipelines and merge their last outputs in 4 coin-wise tensors
        for i in range(self.input_features):
            feature_input = x[:, i].unsqueeze(-1)
            _, (h_n, _) = self.feature_lstms[i](feature_input)
            feature_representations.append(h_n[-1])

        features_stack = torch.stack(feature_representations, dim=1)

        # pass coin tensors into attentions and get 4 attentions outputs
        group_outputs = []
        for i in range(self.num_groups):
            group = features_stack[:, i * self.output_features:(i + 1) * self.output_features]  # [batch, 4, hidden_dim]
            attn_out, _ = self.group_attentions[i](group, group, group)
            group_pooled = attn_out.mean(dim=1)
            group_outputs.append(group_pooled)

        groups_stack = torch.stack(group_outputs, dim=1)

        # apply attention to merged coin pipelines
        final_attn_out, _ = self.final_attention(groups_stack, groups_stack, groups_stack)

        # get the applied attention to the target coin pipeline output
        final_embedding = final_attn_out[:, self.target_coin_index, :]

        output = self.fc(final_embedding)

        output = output.view(batch_size, self.output_features, self.output_window)

        return output
    
    def call(self, x, y):
        return self(x)