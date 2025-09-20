
Report on Hyperparameter Experiments
1. Experiment

This week I experimented with different hyperparameters of a neural network model on the MNIST dataset. The parameters I tested included:

Epochs: ran with 5 and 10 epochs.

Batch size: compared smaller (32), baseline (64), and larger (128) batch sizes.

Model width: varied units1 and units2 between 64, 128, 256.

Model depth: added an additional linear layer with activation.



Observations

Increasing epochs improved stability but gave diminishing returns after ~8 epochs.

Doubling the batch size sped up training but reduced accuracy slightly. Halving the batch size (32) did not improve accuracy compared to the baseline (64).

Adding a third layer increased training time without significantly improving accuracy (stuck at ~85–88%).



2. Reflection

I hypothesized that increasing depth and width would improve accuracy. However, adding a third layer mainly increased computation time without measurable gains, suggesting overparameterization for this dataset. Similarly, larger batch sizes improved efficiency but slightly reduced generalization performance.
From theory:

More epochs usually allow better convergence, but risk overfitting. Here, extra epochs mostly confirmed stability, with no major accuracy gain.

Smaller batch sizes expose the model to noisier gradients, which sometimes helps generalization, but in my runs did not lead to higher accuracy.

Lesson learned: Improvements are not guaranteed by simply scaling up depth or batch size. Optimal hyperparameters depend on balancing training efficiency and generalization.


Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)
