import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self, input_features, dropout, num_layers, hidden_dim):
        super(EncoderGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        outputs, hidden = self.gru(x)
        return hidden

class DecoderGRU(nn.Module):
    def __init__(self, output_features, dropout, num_layers, hidden_dim, teacher_forcing_ratio):
        super(DecoderGRU, self).__init__()
        self.output_features = output_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.gru = nn.GRU(input_size=output_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_features)

    def forward(self, decoder_input, hidden, output_window, target=None):
        outputs = []

        for t in range(output_window):
            # pass input to decoder, do the linear projection and store the prediction
            out, hidden = self.gru(decoder_input, hidden)
            pred = self.fc(out.squeeze(1))
            outputs.append(pred.unsqueeze(1))

            # if we are teacher forcing, pass the target values
            # this way we teach the model to base it's predictions to reality rather than all hallucinated outputs
            if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                decoder_input = pred.unsqueeze(1)

        return torch.cat(outputs, dim=1)

class EncoderDecoderGRU(nn.Module):
    def __init__(self, input_features, output_features, output_window, dropout=0.2, num_layers=3, hidden_dim=128, teacher_forcing_ratio=1.0, target_coin_index=0):
        super(EncoderDecoderGRU, self).__init__()
        self.encoder = EncoderGRU(input_features, dropout=dropout, num_layers=num_layers, hidden_dim=hidden_dim)
        self.decoder = DecoderGRU(output_features, dropout=dropout, num_layers=num_layers, hidden_dim=hidden_dim, teacher_forcing_ratio=teacher_forcing_ratio)
        self.output_window = output_window
        self.output_features = output_features
        self.target_coin_index = target_coin_index

    def forward(self, x, target=None):
        if target is not None:
            target = target.permute(0, 2, 1)

        # get the last data from training, pass it as the decoder's input
        last_x = x[:, -1, :]
        last_x = last_x[:, self.target_coin_index*self.output_features:(self.target_coin_index+1)*self.output_features]
        last_x = last_x.unsqueeze(1) 

        # get encoded output
        hidden = self.encoder(x)
        # pass it to decoder with reshaped target if presence
        
        out = self.decoder(last_x, hidden, self.output_window, target)
        return out.permute(0, 2, 1)

    def set_teacher_forcing_ratio(self, new_value):
        self.decoder.teacher_forcing_ratio = new_value

    def call(self, x, y):
        return self(x, y)