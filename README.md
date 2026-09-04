# Graph neural networks and the energetic cavity method for combinatorial optimization

This is the supporting code for "Graph neural networks and the energetic cavity method for combinatorial optimization" by Joe Bacchus George and George T. Cantwell. It implements Laplacian cut, a standard and spectral GNN, the min-sum algorithm and it's decimated variant, and the min-sum-GNN. Simulated annealing is also provided.

## Contents

This is an outline of the included 

- `data.py` contains functions for creating problem instances and in a batched format that is suitable for training with `torch_geometric`.

- `anneal.py` and `anneal.c` implement simulated annealing. You must first compile the C code according to your device. For MacOS use
    ```zsh
    cc -O3 -march=native -std=c99 -Wall -Wextra -fPIC -shared anneal.c -o libanneal.dylib
    ```
    and for Linux use
    ```zsh
    cc -O3 -march=native -std=c99 -Wall -Wextra -fPIC -shared anneal.c -o libanneal.so
    ```
- `algorithms.py` contains all methods implemented in the paper. Below we provide an outline of the included functions.

We also include a trained model in the folder `models`. This is a MinSumGNN trained for MaxCut on 3-regular graphs.

## Algorithms

In `data.py` the important function is

```python 
make_batch(n_graphs, random_graph, problem, field_noise=1e-15, test=False)
```

**Parameters**:
- `n_graphs`: Number of problem instances to generate.
- `random_graph`: Function to be called that generates random graph topology (`RT`, `RR`, `BA`, `WS`).
- `problem`: Function to be called that generates couplings determining the optimization problem (`MaxCut`, `BinSG`, `EA`, `RFIM`).
- `field_noise`: Additional field noise to encourage non-trivial ground states. _Default: `1e-15`_.
- `test`: Toggle to use if data is used in testing instead of training. _Default: `False`_.

**Returns:** For `test=False`, a list of size `n_graphs` of NetworkX problem instances. For `test=True` a single NetworkX graph containing `n_graphs` problem instances.

> Many subsequent algorithms rely on a `data_generator`. To create this, one could for instance use the `make_batch` function and create
>```python
>data_generator = lambda : make_batch(n_graphs=20, random_graph=RR, problem=MaxCut)
>```

--- 


Here we outline the methods that are usable from `algorithms.py`.

### Simulated Annealing

```python
simulated_annealing(data_generator, sname, n_trials=10, beta_values=10000, beta_final=8)
```

**Parameters**:
- `data_generator`: ata generator containing the graph instances.
- `sname`: Name of the output file.
- `n_trials`: Number of annealing sweeps.
- `beta_values`:
- `beta_final`: 

**Returns:** List of best ground state energies found for each problem instance.

### Laplacian Cut

```python
laplace_maxcut(data_generator, sname)
```

**Parameters**:
- `data_generator`: Data generator containing the graph instances.
- `sname`: Name of the output file.

**Returns:** List of best ground state energies found for each problem instance.

### GNN (Class)

```python
GNN(width=32, depth=256, lap_init=False)
```

**Parameters**:
- `width`: Width of layers, $d$. _Default: `32`_.
- `depth`: Number of layers, $T$. _Default: `256`_.
- `lap_init`: Toggles initialization of weights to implement spectral cut. _Default: `False`_.

**Returns:** Model instance to be used.

### MinSumGNN (Class)

```python
MinSumGNN(width=32, depth=256)
```

**Parameters**:
- `width`: Width of layers, $d$. _Default: `32`_.
- `depth`: Number of layers, $T$. _Default: `256`_.

**Returns:** Model instance to be used.

### Train

```python
train(model, data_generator, fname, epochs=500)
```

**Parameters**:
- `model`: GNN model to be trained.
- `data_generator`: Data generator providing training instances.
- `fname`: Name to be used for the saved model.
- `epochs`: Number of training epochs. _Default: `500`._

**Returns:** Trained model.

### Test

```python
test(model, data_generator, fname, sname, dec=False)
```

**Parameters**:
- `model`: Trained GNN model to evaluate.
- `data_generator`: Data generator providing test instances.
- `fname`: Name of the file containing the trained model.
- `sname`: Name of the output file.
- `dec`: Whether to use decimation during testing. _Default: `False`._

**Returns:** List of best ground state energies found for each problem instance.
