### add frankaemica directory to python path
import sys
import os
script_dir = "cdf/frankaemika"
sys.path.append(script_dir)

import torch

from cdf.frankaemika.mlp import MLPRegression
from nn_cdf import CDF

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
cdf = CDF(device)

# trainer.train_nn(epoches=20000)
model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
model.load_state_dict(torch.load(os.path.join(script_dir,'model_dict.pt'))[49900])
model.to(device)

x = torch.tensor([[0.0, 0.3, 0.3]], device=device, requires_grad=True).float()  # Example obstacle position
q_samples = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], device=device, requires_grad=True).float()  # Example robot configuration
d,grad = cdf.inference_d_wrt_q(x,q_samples,model)
print("Distances:", d)
print("Gradients:", grad)