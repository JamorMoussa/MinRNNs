from min_rnns import MinLSTMCell, MinLSTM
import torch 

B, T, C = 32, 12, 16

H = 10

model = MinLSTM(
    input_size=C, hidden_size=H, num_layers=4
)

print(model)

x = torch.rand(B, T, C)

outs = model(x)

print(outs[0].shape)
print(outs[1].shape)