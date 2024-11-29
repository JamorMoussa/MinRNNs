import torch, torch.nn as nn
import torch.nn.functional as F

from .utils import parallel_scan_log

__all__ = ["MinLSTMCell", "MinLSTM", ]


class MinLSTMCell(nn.Module):

    def __init__(
        self, input_size: int, hidden_size: int 
    ):
        super(MinLSTMCell, self).__init__()

        self.input_size= input_size
        self.hidden_size= hidden_size
        
        self.f_linear = nn.Linear(
            in_features=self.input_size, out_features=self.hidden_size
        )

        self.i_linear = nn.Linear(
            in_features=self.input_size, out_features=self.hidden_size
        )

        self.h_tilde = nn.Linear(
            in_features=self.input_size, out_features=self.hidden_size
        )


    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        
        # x : [batch_size, seq_len, input_size]
        # hx: [batch_size, 1, hidden_size] 

        assert x.ndim >=2, "'x' must be at least of dim=2"

        is_batched = True if x.ndim == 3 else False

        if not is_batched:
            x = x.unsqueeze(0)

        if h0 is None:
            h0 = torch.zeros(x.size(0), 1, self.hidden_size, dtype=x.dtype, device=x.device)


        f_t = F.sigmoid(self.f_linear(x))
        i_t = F.sigmoid(self.i_linear(x))

        h_tilde = self.h_tilde(x)

        fp_t = f_t.div((f_t + i_t + 1e-8))
        ip_t = i_t.div((f_t + i_t + 1e-8))

        h = parallel_scan_log(
            fp_t, 
            torch.cat([
                h0, ip_t * h_tilde
            ], dim=1)
        )

        return h
    

class MinLSTM(nn.Module):

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int
    ):
        super(MinLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        

        self.cell_layers = nn.ModuleList([
            MinLSTMCell(
                input_size= self.input_size if i == 0 else self.hidden_size,
                hidden_size=self.hidden_size
            )
            for i in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):

        # x : [batch_size, seq_len, input_size]
        # h0: [batch_size, num_layers, hidden_size]

        assert x.ndim == 3, "'x' must be of shape of 3 [batch_size, seq_len, input_size]"

        batch_size, _, _ = x.size()

        if h0 is None:
            h0 = torch.zeros(
                batch_size, self.num_layers, self.hidden_size, dtype=x.dtype, device=x.device
            )
        
        h_outs_list = []
        h_t = 0

        for i, cell in enumerate(self.cell_layers):

            h_t = cell(x, h0[:, i, :].unsqueeze(1))
            x = h_t
            
            h_outs_list.append(h_t[:, -1, :].unsqueeze(1)) 

        h_outs = torch.cat(h_outs_list, dim=1)
        outs = h_t 
        return outs, h_outs



