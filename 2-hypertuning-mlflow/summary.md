# Summary week 2

## Observations

For this week i tried to implements new parameters and MLflow to log the results. I put some options for the hyperpatameters in the code, which resulted in a long run time. This is because i tried all the hyperparameters in one time, wich resulted in 2*3*2*3*2*2*2*2 = 256 runs with 6 epochs each. It was a bit hard to find the best hyperparameters in the results, because the MLflow couldnt see my results. ultmately i found my best run to be:
 [32m2025-09-22 14:03:10.299[0m | [1mINFO    [0m | [36mmltrainer.trainer[0m:[36mreport[0m:[36m209[0m - [1mEpoch 5 train 0.2767 test 0.2612 metric ['0.9081'][0m
100%|[38;2;30;71;6m██████████[0m| 6/6 [01:43<00:00, 17.24s/it]

This was a accuracy of 90.81%. This was with the following parameters:
[model]
epochs = 6
metrics = ["Accuracy"]
logdir = "modellogs"
train_steps = 200
valid_steps = 100
reporttypes = ["ReportTypes.MLFLOW", "ReportTypes.TOML"]
[model.optimizer_kwargs]
lr = 0.001
weight_decay = 1e-05

[model.scheduler_kwargs]
factor = 0.1
patience = 10

[model.earlystop_kwargs]
save = false
verbose = true
patience = 10


[types]
epochs = "int"
metrics = "list"
logdir = "WindowsPath"
train_steps = "int"
valid_steps = "int"
reporttypes = "list"
optimizer_kwargs = "dict"
scheduler_kwargs = "dict"
earlystop_kwargs = "dict"


[model]
training = true
input_channels = 1
num_classes = 10
conv_blocks = 4
base_filters = 32
filters_growth = 3.0
kernel_size = 3
pool_kernel = 2
norm = "batch"
hidden_units = [128, 64, 32]
dropout_p = 0.1
global_pool = "adaptive"

I tried to tune it with lower and higher values in the model, but this was the best result.




## Reflection
I had some trouble with my uv.lock file first. After fixing that, i tried using MLflow to log the results. This was not realy smooth, and costed me a lot of time.
After setting it all up, it didnt show me any information.
So i just tried to code some hyperparameter testing.

In class i have to ask Raoul about the mlflow link, so i can review my tests.

Find the [instructions](./instructions.md)

[Go back to Homepage](../README.md
