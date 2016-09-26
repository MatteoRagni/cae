# CAE

## Binary `cae-bin`

```
usage: caes-bin [-h] [-id IDENTIFIER] -w WORKSPACE -d DATASET -m MODEL
                [--training-id TRAINING_ID] [-s SAVE_FILE] [-l LOAD_FILE]
                [-bs BATCH_SIZE] [-sz STEPS] [-bb BATCH_BLOCK]
                [--learning-rate LEARN_RATE] [--residual-learning]

Stacked Convolutional Autoencoder

optional arguments:
  -h, --help            show this help message and exit
  -id IDENTIFIER, --identifier IDENTIFIER
                        Training identifier - time base [20160926-1511]
  -w WORKSPACE, --workspace WORKSPACE
                        Workspace directory. Will contain run files generated
                        during training and inference
  -d DATASET, --dataset DATASET
                        Dataset directory. Must contain training and inference
                        datasets
  -m MODEL, --model MODEL
                        Model binary file
  --training-id TRAINING_ID
                        Run directory will be created inside workspace
  -s SAVE_FILE, --save SAVE_FILE
                        Checkpoint saving file
  -l LOAD_FILE, --load LOAD_FILE
                        Checkpoint loading file (will skip training)
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size (number of examples for learning step)
  -sz STEPS, --steps STEPS
                        Number of reiteratios on a single batch
  -bb BATCH_BLOCK, --batch-block BATCH_BLOCK
                        Number of blocks of batches to be loaded (1 batch =
                        1000 figures)
  --learning-rate LEARN_RATE
                        Learning rate hyper-parameters
  --residual-learning   Enable residual learning (NO -> y = f(g(x)), YES -> y
                        = f(g(x)) + x) [NO]

Matteo Ragni, David Windridge, Paolo Bosetti - 2016
```
