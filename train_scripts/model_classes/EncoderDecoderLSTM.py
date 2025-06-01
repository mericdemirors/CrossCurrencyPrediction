import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, input_features, dropout, num_layers, hidden_dim):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        outputs, (hidden, cell) = self.lstm(x)

        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, output_features, dropout, num_layers, hidden_dim, teacher_forcing_ratio):
        super(DecoderLSTM, self).__init__()
        self.output_features = output_features
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.lstm = nn.LSTM(input_size=output_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=self.output_features)

    def forward(self, decoder_input, hidden, cell, output_window, target=None):
        outputs = []

        for t in range(output_window):
            # pass input to decoder, do the linear projection and store the prediction
            out, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            pred = self.fc(out.squeeze(1))
            outputs.append(pred.unsqueeze(1))

            # if we are teacher forcing, pass the target values
            # this way we teach the model to base it's predictions to reality rather than all hallucinated outputs
            if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                decoder_input = pred.unsqueeze(1)
        
        return torch.cat(outputs, dim=1)

class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_features, output_features, output_window, dropout=0.2, num_layers=3, hidden_dim=128, teacher_forcing_ratio=1.0, target_coin_index=0):
        super(EncoderDecoderLSTM, self).__init__()
        self.encoder = EncoderLSTM(input_features=input_features, dropout=dropout, num_layers=num_layers, hidden_dim=hidden_dim)
        self.decoder = DecoderLSTM(output_features=output_features, dropout=dropout, num_layers=num_layers, hidden_dim=hidden_dim, teacher_forcing_ratio=teacher_forcing_ratio)
        self.output_features = output_features
        self.output_window = output_window
        self.target_coin_index = target_coin_index
        self.teacher_forcing_ratio = self.decoder.teacher_forcing_ratio

    def forward(self, x, target=None):
        if target is not None:
            target = target.permute(0, 2, 1)
        
        # get the last data from training, pass it as the decoder's input
        last_x = x[:, -1, :]
        last_x = last_x[:, self.target_coin_index*self.output_features:(self.target_coin_index+1)*self.output_features]
        last_x = last_x.unsqueeze(1)

        # get encoded output
        hidden, cell = self.encoder(x)
        # pass it to decoder with reshaped target if presence

        out = self.decoder(last_x, hidden, cell, self.output_window, target)
        out = out.permute(0, 2, 1)
        return out
    
    def set_teacher_forcing_ratio(self, new_value):
        self.decoder.teacher_forcing_ratio = new_value

    def call(self, x, y):
        return self(x, y)