import itertools
import os
import sys
import warnings
import argparse

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # isort:skip deterministic PyTorch

import numpy as np  # noqa: E402
import torch as T  # noqa: E402

from src import pipeline  # noqa: E402
from src.run.pipeline import run_id  # noqa: E402
from src.ds.causal_graph import sample_cg, CausalGraph

warnings.filterwarnings(
    "ignore", "Setting attributes on ParameterDict is not supported.")
warnings.filterwarnings("ignore", category=UserWarning,
                        module='pytorch_lightning.utilities.distributed')

T.set_deterministic(True)

priority_graphs = ["napkin"]
valid_graphs = {"m", "backdoor", "frontdoor", "bow", "iv", "bad_m_2", "extended_bow"}

test_graphs = {"simple", "bad_m", "bdm", "chain", "double_bow"}

parser = argparse.ArgumentParser(description="ID Experiment Runner")
parser.add_argument('name', help="name of the experiment")

parser.add_argument('--check-progress', action="store_true",
                    help="outputs % complete (approximate)")

parser.add_argument('--graph', '-G', help="name of preset graph")
parser.add_argument('--n-vars', type=int, default=4, help="number of variables (default: 4)")
parser.add_argument('--p-dir', type=float, default=0.4,
                    help="probability of directed edge in sampled graphs (default: 0.4)")
parser.add_argument('--p-bidir', type=float, default=0.2,
                    help="probability of bidirected edge in sampled graphs (default: 0.2)")
parser.add_argument('--enforce-dir-path', action="store_true",
                    help="sampled graphs must have directed path from X to Y")
parser.add_argument('--enforce-bidir-path', action="store_true",
                    help="sampled graphs must have bidirected path from X to Y")

parser.add_argument('--n-trials', '-t', type=int, default=1, help="number of trials")
parser.add_argument('--n-samples', '-n', type=int, default=10000, help="number of samples (default: 10000)")
parser.add_argument('--dim', '-d', type=int, default=1, help="dimensionality of variables (default: 1)")
parser.add_argument('--n-epochs', type=int, default=1000, help="number of epochs (default: 1000)")
parser.add_argument('--gpu', help="GPU to use")

parser.add_argument('--n-resample-trials', '-r', type=int, default=1,
                    help="number of times the same trial is rerun (default: 1)")

args = parser.parse_args()

assert args.graph is None or args.graph == "all" or args.graph in valid_graphs \
       or args.graph in test_graphs or args.graph in priority_graphs

gpu_used = 0 if args.gpu is None else args.gpu

if args.graph is not None:
    if args.graph == "all":
        graph_set = priority_graphs + list(valid_graphs)
    else:
        graph_set = {args.graph}

    for graph in graph_set:
        for i in range(args.n_trials):
            while True:
                cg = CausalGraph.read("dat/cg/{}.cg".format(graph))
                cg_args = graph
                try:
                    graph_path = "{}/{}".format(args.name, graph)
                    if not run_id(graph_path, cg, cg_args, args.n_samples, args.dim, args.n_epochs, i,
                                  args.n_resample_trials, gpu=gpu_used):
                        break
                except Exception as e:
                    print(e)
                    print('[failed]', i, args.name)
                    raise
else:
    cg_args = {
        "n_vars": args.n_vars,
        "p_dir": args.p_dir,
        "p_bidir": args.p_bidir,
        "enf_dir": args.enforce_dir_path,
        "enf_bidir": args.enforce_bidir_path
    }
    for enf_ID in [False, True]:
        for i in range(args.n_trials):
            while True:
                cg = sample_cg(args.n_vars, args.p_dir, args.p_bidir, enforce_direct_path=args.enforce_dir_path,
                           enforce_bidirect_path=args.enforce_bidir_path, enforce_ID=enf_ID)
                try:
                    id_path = "{}/ID".format(args.name) if enf_ID else "{}/nonID".format(args.name)
                    if not run_id(id_path, cg, cg_args, args.n_samples, args.dim, args.n_epochs, i,
                                  args.n_resample_trials, gpu=gpu_used):
                        break
                except Exception as e:
                    print(e)
                    print('[failed]', i, args.name)
                    raise
