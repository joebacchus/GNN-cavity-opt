import torch
import networkx as nx
from algorithms import *
from data import make_batch, RR, WS, BA, MaxCut, RFIM, DEV

model = NodeMinSumGNN(width=32, depth=256).to(DEV)
model.load_state_dict(torch.load("models/NodeMinSumGNN_RFIM_WS.pth", map_location=DEV))
model.eval()

datas = make_batch(10, lambda: WS(8192), RFIM, test=True)

for data in datas:
    sigma = spins(model, data, n_trials=10)
    H = energy(sigma, data)
    best = H.min().item()
    print(f"H/n = {best:.4f}") 

