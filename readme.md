# Neural Causal Models

To run the identification experiments, navigate to the base directory of the repository and run

`python -m src.experiment.experiment1 Experiment1 -G all -t 20 --n-epochs 3000 -r 4`

Once completed, generate plots by running

`python -m src.experiment.experiment1_results out/IDExperiments/Experiment1`

Plots will appear in `out/IDExperiments/Experiment1/figs`.

To run the estimation experiments, navigate to the base directory of the repository and run

`python -m src.experiment.experiment2 BiasedNLLNCMPipeline`

Once completed, run the other estimation methods with

`python -m src.experiment.exp2_estimation_runner out/BiasedNLLNCMPipeline`

Finally, to generate plots, run

`python -m src.experiment.experiment2_results`

Plots will appear in the `img` directory inside the base directory.
