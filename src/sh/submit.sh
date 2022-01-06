#!/bin/bash
#
# Generate experiment parameterizations and submit corresponding jobs.
#

src/sh/generate_experiments.sh > out/experiments.txt
while read line; do sbatch src/sh/job.sh $line; done < out/experiments.txt
