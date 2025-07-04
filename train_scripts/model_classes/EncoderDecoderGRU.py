import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self, input_features, dropout, num_layers, hidden_dim):
        super(EncoderGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        outputs, hidden = self.gru(x)
        return hidden

class DecoderGRU(nn.Module):
    def __init__(self, output_features, dropout, num_layers, hidden_dim, teacher_forcing_ratio):
        super(DecoderGRU, self).__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.gru = nn.GRU(input_size=output_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_features)

    def forward(self, decoder_input, hidden, output_window, target):
        outputs = []

        for t in range(output_window):
            # pass input to decoder, do the linear projection and store the prediction
            out, hidden = self.gru(decoder_input, hidden)
            pred = self.fc(out.squeeze(1))
            outputs.append(pred.unsqueeze(1))

            # if we are teacher forcing, pass the target values
            # this way we teach the model to base it's predictions to reality rather than all hallucinated outputs
            if torch.rand(1) < self.teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                decoder_input = pred.unsqueeze(1)
        return torch.cat(outputs, dim=1)

class EncoderDecoderGRU(nn.Module):
    def __init__(self, input_features, output_features, output_window, dropout, num_layers, hidden_dim, teacher_forcing_ratio, output_col_indices_in_input_cols):
        super(EncoderDecoderGRU, self).__init__()
        self.encoder = EncoderGRU(input_features=input_features, dropout=dropout, num_layers=num_layers, hidden_dim=hidden_dim)
        self.decoder = DecoderGRU(output_features=output_features, dropout=dropout, num_layers=num_layers, hidden_dim=hidden_dim, teacher_forcing_ratio=teacher_forcing_ratio)
        self.output_features = output_features
        self.output_window = output_window
        self.output_col_indices_in_input_cols = output_col_indices_in_input_cols
        self.teacher_forcing_ratio = self.decoder.teacher_forcing_ratio

    def forward(self, x, target):
        x = x.permute(0, 2, 1)
        target = target.permute(0, 2, 1)

        # get the last data from training, pass it as the decoder's input
        last_x = x[:, -1, :]
        last_x = last_x[:,self.output_col_indices_in_input_cols]
        last_x = last_x.unsqueeze(1) 

        # get encoded output
        hidden = self.encoder(x)
        # pass it to decoder with reshaped target if presence
        
        out = self.decoder(last_x, hidden, self.output_window, target)
        out = out.permute(0, 2, 1)
        return out

    def set_teacher_forcing_ratio(self, new_value):
        self.decoder.teacher_forcing_ratio = new_value

    def call(self, x, y):
        return self(x, y)