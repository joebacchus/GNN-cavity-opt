import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_recursive_tree(n):
    parents = np.array([np.random.randint(i) for i in range(1, n)])
    G = nx.empty_graph(n)
    G.add_edges_from(zip(parents, range(1, n)))
    return G

# random graph models:
RT = lambda n : random_recursive_tree(n)
RR = lambda n : nx.random_regular_graph(3, n)
BA = lambda n : nx.barabasi_albert_graph(n, 2)
WS = lambda n : nx.watts_strogatz_graph(n, 4, 0.1)

# optimization problems:
MaxCut = (lambda : -1.0,                       lambda : 0.0)
BinSG  = (lambda : np.random.choice([-1,+1]),  lambda : 0.0)
EA     = (lambda : np.random.normal(),         lambda : 0.0)
RFIM   = (lambda : 1.0,                        lambda : np.random.normal())

def make_batch(n_graphs, random_graph, problem, field_noise=1e-15, test=False):
    if test:
        datas = []
        for _ in range(n_graphs):
            G = random_graph()
            for i in G.nodes():
                G.nodes[i]['weight'] = problem[1]()
            for i, j in G.edges():
                G.edges[i, j]['weight'] = problem[0]()
            datas.append(data_from_networkx(G, device=DEV, field_noise=field_noise))
        return datas
    else:
        G = nx.disjoint_union_all([random_graph() for _ in range(n_graphs)])
        for i in G.nodes():
            G.nodes[i]['weight'] = problem[1]()
        for i, j in G.edges():
            G.edges[i, j]['weight'] = problem[0]() 
        return data_from_networkx(G, device=DEV, field_noise=field_noise)

def data_from_networkx(G, device, field_noise=1e-15):
    """Build a PyG Data object carrying everything MinSum-GNN needs.

    Convention: messages[(i, j)] is the index of ν_{i<-j} (message arriving at
    i from sender j). Each undirected edge therefore yields two messages.

      edge_index    [2, M]   for each message (i<-j): row 0 = i, row 1 = j
      message_index [2, K]   pairs (target ν_{i<-j}, source ν_{j<-k}), k ≠ i
      marg_index    [2, ·]   pairs (node i, incoming message ν_{i<-j})
      h_i, J, message_J, prior — fields and couplings on each graph.
    """
    edges = sorted(((i, j) for e in G.edges() for i, j in (e, e[::-1])),
                   key=lambda x: (x[0], x[1]))
    messages = {e: idx for idx, e in enumerate(edges)}

    h = {i: G.nodes[i]['weight'] + field_noise * np.random.normal() for i in G.nodes()}

    # Message update: ν_{i<-j} ← ... + Σ_{k ∈ N(j)\i} minsum(J_{jk}, ν_{j<-k})
    message_src, message_dst, message_J = [], [], []
    for (i, j) in edges:
        for k in sorted(G.neighbors(j)):
            if k != i:
                message_src.append(messages[(i, j)]) # target ν_{i<-j}
                message_dst.append(messages[(j, k)]) # source ν_{j<-k}
                message_J.append(G.edges[j, k]['weight'])

    # Marginal at node i: sums its incoming messages ν_{i<-j} for j ∈ N(i)
    marg_src, marg_dst, h_i, J = [], [], [], []
    for i in sorted(G.nodes()):
        h_i.append(h[i])
        for j in sorted(G.neighbors(i)):
            marg_src.append(i)
            marg_dst.append(messages[(i, j)]) # incoming ν_{i<-j}
            J.append(G.edges[i, j]['weight'])

    src, dst = zip(*edges)
    data = Data().to(device)
    data.G = G
    data.number_of_nodes = G.number_of_nodes()
    data.messages = messages
    data.edge_index    = torch.tensor([src, dst],                 dtype=torch.long, device=device)
    data.message_index = torch.tensor([message_src, message_dst], dtype=torch.long, device=device)
    data.marg_index    = torch.tensor([marg_src, marg_dst],       dtype=torch.long, device=device)
    data.h_i       = torch.tensor(h_i,       dtype=torch.float32, device=device)
    data.J         = torch.tensor(J,         dtype=torch.float32, device=device)
    data.message_J = torch.tensor(message_J, dtype=torch.float32, device=device)
    data.prior     = torch.zeros((1, len(h_i)), dtype=torch.float32, device=device)
    return data