import sys
import os
import json
import argparse
import numpy as np
import torch as T
from src.run.pipeline import run_other_estimators


parser = argparse.ArgumentParser(description="Alternate Estimation Method Runner")
parser.add_argument('directory', help="directory for estimations")

parser.add_argument('--cap', type=int, default=-1, help="maximum trial number to check")
parser.add_argument('--graph', '-G', default="all", help="if entered, only estimates in cases for specific graph")
parser.add_argument('--redo-werm', action="store_true", help="redo werm calculation even if it already exists")
parser.add_argument('--redo-naive', action="store_true", help="redo naive NN calculation even if it already exists")
parser.add_argument('--gpu', help="GPU to use")

args = parser.parse_args()

d = args.directory

if args.gpu is None:
    dev = "cpu"
else:
    dev = T.device("cuda:{}".format(args.gpu))


def get_params(dir_name):
    param_pairs = dir_name.split('-')
    params = dict()
    for pair in param_pairs:
        pair_split = pair.split('=')
        params[pair_split[0]] = pair_split[1]
    return params


for item in os.listdir(d):
    params = get_params(item)
    if args.cap > 0 and int(params["trial_index"]) >= args.cap:
        continue

    if args.graph != "all" and params["graph"] != args.graph:
        continue

    redo_list = []
    if args.redo_werm:
        redo_list.append("werm")
    if args.redo_naive:
        redo_list.append("naive")
    new_dir = "{}/{}".format(d, item)
    run_other_estimators(new_dir, item, dev, redo_list)
    