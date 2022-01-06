#!/bin/bash
#
# Submit all experiments as jobs using sbatch.
#

tmp=$(mktemp -d)
trap 'rm -rf $tmp' EXIT

seq 0 9 > $tmp/--trial-index
python -c \
    'import numpy as np;'`
    `'print("\n".join(map(str, list((10 ** np.linspace(3, 6, 31)).astype(int)))))' > $tmp/--n-samples
echo "$(dirname $0)/../../dat/cg/"{simple,backdoor,frontdoor,napkin,m,bow,iv}.cg |
    tr ' ' '\n' |
    xargs -n1 readlink -f > $tmp/--graph

function merge {
    f=$1
    shift

    if [ -z "$f" ]; then
        return
    elif [ $# -ge 1 ]; then
        join -j 9999999 <(sed -- "s/^/$f=/" "$f") <(merge $@)
    else
        sed -- "s/^/$f=/" "$f"
    fi
}

cd $tmp && merge --trial-index --n-samples --graph |
    sed 's/^/python -m src.run.pipeline/'
