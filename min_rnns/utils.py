import torch 
import torch.nn.functional as F 

def parallel_scan_log(log_coeffs, log_values):

    # log_coeffs: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)

    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    
    log_h0_plus_b_star = torch.logcumsumexp(
        log_values - a_star, dim=1
    )
    
    log_h = a_star + log_h0_plus_b_star
    
    return torch.exp(log_h)[:, 1:]