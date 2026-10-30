import torch
import networkx as nx
from algorithms import *
from data import make_batch, RR, MaxCut, DEV

model = NodeMinSumGNN(width=32, depth=256).to(DEV)
model.load_state_dict(torch.load("models/NodeMinSumGNN_MaxCut_RR.pth", map_location=DEV))
model.eval()

datas = make_batch(2, lambda: RR(8192), MaxCut, test=True)

for data in datas:
    sigma = spins(model, data, n_trials=10)
    H = energy(sigma, data)
    best = H.min().item()
    print(f"H/n = {best:.4f}") 

