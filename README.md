# CAE

## Binary `cae-bin`

```
usage: cae-bin [-h] -in DATASET [-gpu] [-r RUNS] [-s SAVE] [-bs BATCHSIZE]
               [-ss STEPSIZE] [-lr LEARNRATE] [-v VERB] [-tg]

Stacked Convolutional Autoencoder

optional arguments:
  -h, --help            show this help message and exit
  -in DATASET           Input dataset for the training
  -gpu                  Enable CUDA and CuDNN [NO]
  -r RUNS, --run RUNS   Define a specific run directory, inside the training
                        directory [/tmp/trinaing_nn/20160609-1607-run]
  -s SAVE, --save SAVE  Define save positions for the result of the training,
                        as checkpoint
                        [/tmp/trinaing_nn/20160609-1607-save/cae.ckpt]
  -bs BATCHSIZE, --batchsize BATCHSIZE
                        Define the size of each batch. It must be compatible
                        with RAM or VRAM capabilities. It must be positive
                        [10]
  -ss STEPSIZE, --stepsize STEPSIZE
                        Define the number of step optimizer will run on each
                        batch. It must be positive [10]
  -lr LEARNRATE, --learnrate LEARNRATE
                        Define the learning rate for the optimizer [0.010000]
  -v VERB, --verbosity VERB
                        Define verbosity level, from 0 to 3 [1]
  -tg, --telegram       Enable notifications using system telegram bot [NO]

Matteo Ragni, David Windridge - 2016

```
