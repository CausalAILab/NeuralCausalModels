# Neural Causal Models

This repository contains the code for the paper ["The Causal-Neural Connection: Expressiveness, Learnability, and Inference"](https://arxiv.org/abs/2107.00793) by Kevin Xia, Kai-Zhan Lee, Yoshua Bengio, and Elias Bareinboim.

Disclaimer: This code is offered publicly with the MIT License, meaning that others can use it for any purpose, but we are not liable for any issues arising from the use of this code. Please cite our work if you found this code useful.

## Setup

```
python -m pip install -r requirements.txt
```

## Running the code

To run the identification experiments, navigate to the base directory of the repository and run

```
python -m src.experiment.experiment1 Experiment1 -G all -t 20 --n-epochs 3000 -r 4
```

Once completed, generate plots by running

```
python -m src.experiment.experiment1_results out/IDExperiments/Experiment1
```

Plots will appear in `out/IDExperiments/Experiment1/figs`.

To run the estimation experiments, navigate to the base directory of the repository and run

```
python -m src.experiment.experiment2 BiasedNLLNCMPipeline
```

Once completed, run the other estimation methods with

```
python -m src.experiment.exp2_estimation_runner out/BiasedNLLNCMPipeline
```

Finally, to generate plots, run

```
python -m src.experiment.experiment2_results
```

Plots will appear in the `img` directory inside the base directory.

## Graphically identifying L2 queries with our code

The following example code will identify $P(y | do(x))$ in the napkin graph when run from the root directory of our project.

```python
from src.ds import CausalGraph

# feel free to use any of the causal graph (.cg) files in dat/cg
cg = CausalGraph.read('dat/cg/napkin.cg')

# equivalently, construct the graph yourself; the graph below is the napkin graph, which does not have a back-door admissible set
cg = CausalGraph(
    V=('X', 'Y', 'W_1', 'W_2'),
    directed_edges=[('W_1', 'W_2'), ('W_2', 'X'), ('X', 'Y')],
    bidirected_edges=[('X', 'W_1'), ('W_1', 'Y')]
)

# identifies P(y | do(x)); note that the inputs to the identify algorithm are sets of variables! in this case, they are singletons.
print(cg.identify({'X'}, {'Y'}))
print(cg.identify({'X'}, {'Y'}).get_latex())
# output: [sum{W_1}[P(W_1)P(Y,W_1,W_2,X) / P(W_1,W_2)] / sum{W_1}[P(W_1)P(W_1,W_2,X) / P(W_1,W_2)]]
# output: \left[\frac{\sum_{W_1}\left[\frac{P(W_1)P(Y,W_1,W_2,X)}{P(W_1,W_2)}\right]}{\sum_{W_1}\left[\frac{P(W_1)P(W_1,W_2,X)}{P(W_1,W_2)}\right]}\right]
```

The outputted LaTeX code corresponds to the following expression:

$$P(Y | do(X)) = \left[\frac{\sum_{W_1}\left[\frac{P(W_1)P(Y,W_1,W_2,X)}{P(W_1,W_2)}\right]}{\sum_{W_1}\left[\frac{P(W_1)P(W_1,W_2,X)}{P(W_1,W_2)}\right]}\right]$$
