#!/bin/bash
#
# Run the Python script using a single thread or multiple threads.
#

cd "$(dirname ${BASH_SOURCE[0]})/../.."
export PYTHONPATH=$PYTHONPATH:$PWD
if [ "$1" = 'test' ]; then
    python -m src.main NLLNCMPipeline 0 2>&1
else
    tmux new-session -d "python -m src.run.monitor"
    tmux move-window -s 0:0 -t 0:8
    for i in {0..7}; do
        tmux new-window -t 0:$i "bash --init-file <(echo 'python -m src.main NLLNCMPipeline $i 2>&1; . $HOME/.bashrc')"
    done
fi
