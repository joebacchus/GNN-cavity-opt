"""
Reference implementation of MinSum-GNN.

Notation (matches paper):
  σ_i ∈ {±1}     spins
  h_i ∈ ℝ        local fields
  J_ij ∈ ℝ       edge couplings (data.J on the node graph; data.message_J on the message graph)
  ν_{i<-j} ∈ ℝ   min-sum BP messages on directed edges
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import json 
import os
import scipy.sparse as sp
import networkx as nx
import numpy as np
from tqdm import tqdm
from anneal import parallel_anneal

weight_initializer = lambda W : None

# ---- Energy & spin readout ------------------------------------------------

def energy(sigma, data):
    """H/n = -(1/n)[½ Σ_ij J_ij σ_i σ_j + Σ_i h_i σ_i],  σ shape [T, n]."""
    src, dst = data.edge_index
    pair  = (data.J * sigma[:, src] * sigma[:, dst]).sum(-1) / 2  
    field = (data.h_i * sigma).sum(-1)
    return -(pair + field) / data.number_of_nodes

@torch.no_grad()
def spins(model, data, n_trials=1):
    """Run model and threshold to ±1 (sign(0)=+1)."""
    return torch.where(model(data, n_trials) >= 0, 1.0, -1.0)
        
# ---- Simulated Annealing ----------------------------------------------------

def simulated_annealing(data_generator, sname, n_trials=10, beta_values=10000, beta_final=8):
    # if os.path.exists(sname):
    #     return
    datas = data_generator()
    minH = np.zeros(len(datas))
    for idx, data in tqdm(enumerate(datas), desc="graph number", leave=False):
        sigma = parallel_anneal(data.G, beta_final, int(beta_values), n_trials)
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=data.h_i.device)
        H = energy(torch.where(sigma >= 0, 1.0, -1.0), data)
        minH[idx] = min(H.detach().cpu().tolist())
    os.makedirs(os.path.dirname(sname), exist_ok=True)
    with open(sname, "w") as f:
        json.dump(list(minH), f)

# ---- Spectral Laplacian Maximum Cut ---------------------------------------

def laplace_maxcut(data_generator, sname):
    # if os.path.exists(sname):
    #     return
    datas = data_generator()
    minH = np.zeros(len(datas))
    for idx, data in tqdm(enumerate(datas), desc="graph number", leave=False):
        L = nx.laplacian_matrix(data.G, nodelist=range(data.number_of_nodes))
        L = sp.diags(1.0 / L.diagonal()) @ L
        eigvec = sp.linalg.eigs(L, k=1, which='LM')[1].flatten().real
        eigvec = np.sign(eigvec)
        eigvec[eigvec == 0] = 1
        sigma = torch.as_tensor(eigvec, dtype=torch.float32, device=data.h_i.device)
        sigma = torch.where(sigma >= 0, 1.0, -1.0).unsqueeze(0)
        H = energy(sigma, data)
        minH[idx] = min(H.detach().cpu().tolist())
    os.makedirs(os.path.dirname(sname), exist_ok=True)
    with open(sname, "w") as f:
        json.dump(list(minH), f)


# ---- Min-sum operator -----------------------------------------------------

def minsum(J, nu):
    """J ⊕ ν = sign(J·ν) · min(|J|, |ν|)."""
    return (J * nu).sign() * torch.minimum(J.abs(), nu.abs())

# ---- Sum-prod operator -----------------------------------------------------

def sumprod(J, nu):
    """J ⊕ ν = arctanh(tanh(J) · tanh(ν))."""
    return torch.arctanh(J.tanh() * nu.tanh())

# ---- Generic GNN layer ----------------------------------------------------

class GNNLayer(MessagePassing):
    """y_i = b + W₁ x_i + W₂ · mean_j J_ij x_j + C h_i."""
    def __init__(self, in_dim, out_dim, h_dim):
        super().__init__(aggr="mean")
        self.W1 = nn.Linear(in_dim, out_dim, bias=True)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.C  = nn.Linear(h_dim,  out_dim, bias=False)
        for L in (self.W1, self.W2, self.C):
            weight_initializer(L.weight)
        nn.init.zeros_(self.W1.bias)

    def forward(self, x, h, edge_index, J):
        agg = self.propagate(edge_index, x=x, J=J)
        return self.W1(x) + self.W2(agg) + self.C(h)

    def message(self, x_j, J):
        return J.view(-1, 1) * x_j


# ---- Vanilla GNN baseline -------------------------------------------------

class GNN(nn.Module):
    def __init__(self, width, depth, lap_init=False):
        super().__init__()
        self.width = width
        self.layers = nn.ModuleList(GNNLayer(width, width, h_dim=1) for _ in range(depth))
        self.output = nn.Linear(width, 1, bias=False)
        if lap_init:
            self._lap_init()

    @torch.no_grad()
    def _lap_init(self):
        """Pin channel 0 to y_i[0] ← x_i[0] − mean_j x_j[0] (random-walk Laplacian)."""
        self.output.weight[0] = 0; self.output.weight[0, 0] = 1
        for layer in self.layers:
            layer.W1.weight[0] = 0; layer.W1.weight[0, 0] = +1; layer.W1.bias[0] = 0
            layer.W2.weight[0] = 0; layer.W2.weight[0, 0] = +1
            layer.C.weight[0]  = 0

    def forward(self, data, n_trials=1):
        n = data.number_of_nodes
        y = torch.randn(n_trials, n, self.width, device=data.h_i.device)
        h = data.h_i.unsqueeze(-1) # [n, 1]
        for layer in self.layers:
            y = torch.tanh(layer(y, h, data.edge_index, data.J))
        return torch.tanh(self.output(y)).squeeze(-1) # [T, n]


# ---- MinSum + GNN hybrid --------------------------------------------------

class MinSumGNN(nn.Module):
    """Each iteration: (1) BP message update ν,
                       (2) GNN layer over the message graph,
                       (3) learned mix of (ν, hidden).
    Final readout passes through `update_marg` to get node marginals."""

    def __init__(self, width, depth):
        super().__init__()
        self.width, self.depth = width, depth
        self.layers = nn.ModuleList(GNNLayer(width, width, h_dim=3) for _ in range(depth))
        self.agg    = nn.ModuleList(nn.Linear(width+1, width+1, bias=False) for _ in range(depth))
        self.output = nn.Linear(width+1, 1, bias=False)
        for L in [*self.agg, self.output]:
            weight_initializer(L.weight)
        self._pin_minsum_channel()

    @torch.no_grad()
    def _pin_minsum_channel(self):
        """Channel 0 of every `agg` and `output` is identity → the scalar BP
        message ν flows through the network untouched by the learned weights."""
        for L in [*self.agg, self.output]:
            L.weight[0] = 0; L.weight[0, 0] = 1

    def update_message(self, data, T, nu):
        """ν_{i<-j} ← h_j + prior_j + Σ_{k∈N(j)\\i} minsum(J_{jk}, ν_{j<-k})."""
        agg = torch.zeros_like(nu).scatter_reduce(
            1, data.message_index[0].unsqueeze(0).expand(T, -1),
            minsum(data.J, nu)[:, data.message_index[1]], reduce="sum")
        nu_new = data.prior[:, data.edge_index[1]] + data.h_i[data.edge_index[1]] + agg
        return 0.5 * nu + 0.5 * nu_new # damping

    def update_marg(self, data, T, tau):
        """marg_i ← h_i + prior_i + Σ_{j∈N(i)} minsum(J_ij, τ_{i<-j})."""
        n = data.number_of_nodes
        agg = torch.zeros((T, n), device=data.h_i.device).scatter_reduce(
            1, data.marg_index[0].unsqueeze(0).expand(T, -1),
            minsum(data.J, tau)[:, data.marg_index[1]], reduce="sum")
        return data.prior + data.h_i + agg

    def forward(self, data, n_trials=1):
        T, M, dev = n_trials, len(data.messages), data.h_i.device
        nu = torch.randn(T, M, device=dev)
        y  = torch.ones(T, M, self.width, device=dev)
        h  = torch.stack([data.h_i[data.edge_index[0]], # [M, 3]
                          data.h_i[data.edge_index[1]],
                          data.J], dim=-1)
        for l in range(self.depth):
            nu = self.update_message(data, T, nu)
            y  = torch.tanh(self.layers[l](y, h, data.message_index, data.message_J))
            nu, y = torch.split(self.agg[l](torch.cat([nu.unsqueeze(-1), y], dim=-1)),
                                [1, self.width], dim=-1)
            nu = nu.squeeze(-1)
        tau = self.output(torch.cat([nu.unsqueeze(-1), y], dim=-1)).squeeze(-1)
        return self.update_marg(data, T, tau)

# ---- Decimation -----------------------------------------------------------

@torch.no_grad()
def decimate(model, data, n_trials, dec_frac=0.05, strength=1000.0):
    """Run `n_trials` parallel decimation chains. Each round, every chain
    freezes its own random `dec_frac · n` nodes to their current sign by
    raising the prior to ±strength."""
    n   = data.number_of_nodes
    dev = data.h_i.device
    data.prior = torch.zeros((n_trials, n), device=dev)
    k = max(1, int(n * dec_frac))
    perm = torch.argsort(torch.rand(n_trials, n, device=dev), dim=1) # [T, n]
    for start in range(0, n, k):
        idx = perm[:, start:min(start + k, n)] # [T, k']
        data.prior.scatter_(1, idx, torch.gather(spins(model, data, n_trials), 1, idx) * strength)
    sigma = spins(model, data, n_trials)
    data.prior = torch.zeros((1, n), device=dev)
    return sigma

# ---- Training -------------------------------------------------------------

def train(model, data_generator, fname, n_steps = 1000):
    if os.path.exists(fname):
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.99 
    )
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    for step in tqdm(range(n_steps), desc="training_step", leave=False):
        optimizer.zero_grad()
        data = data_generator()
        z = torch.tanh(model(data, n_trials=1))
        loss = torch.sum(energy(z, data))
        H = energy(torch.where(z >= 0, 1.0, -1.0), data)[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
    torch.save(model.state_dict(), fname)
    
# ---- Testing -------------------------------------------------------------

@torch.no_grad()
def test(model, data_generator, fname, sname, dec=False):
    model.load_state_dict(torch.load(fname)) if fname else None
    model.eval()
    datas = data_generator()
    minH = np.zeros(len(datas))
    for idx, data in tqdm(enumerate(datas), desc="graph number", leave=False):
        if dec:
            z = decimate(model, data, n_trials=10)
        else:
            z = torch.tanh(model(data, n_trials=10))
        H = energy(torch.where(z >= 0, 1.0, -1.0), data)
        minH[idx] = min(H.detach().cpu().tolist())
    os.makedirs(os.path.dirname(sname), exist_ok=True)
    with open(sname, "w") as f:
        json.dump(list(minH), f)