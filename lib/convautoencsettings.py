#! /usr/bin/env python3

from six.moves import cPickle as pickle
import cmd
import autoencoder

#   ___           __ _                    _   _
#  / __|___ _ _  / _(_)__ _ _  _ _ _ __ _| |_(_)___ _ _
# | (__/ _ \ ' \|  _| / _` | || | '_/ _` |  _| / _ \ ' \
#  \___\___/_||_|_| |_\__, |\_,_|_| \__,_|\__|_\___/_||_|
#                     |___/
# Do not edit above this line

OUTPUT_FILE       = "autoencoder.pickle"
BATCH_SIZE        = 10
RESIDUAL_LEARNING = False

sets1 = autoencoder.ConvAutoEncSettings()
sets1.prefix_name = "alpha"
sets1.input_shape = [BATCH_SIZE, 161, 161, 1]
sets1.corruption = False
sets1.layers = 1
sets1.patch_size = 5
sets1.strides = [1, 2, 2, 1]
sets1.padding = 'SAME'
sets1.depth_increasing = 4
sets1.residual_learning = RESIDUAL_LEARNING

sets2 = autoencoder.ConvAutoEncSettings()
sets2.prefix_name = "beta"
sets2.input_shape = [BATCH_SIZE, 81, 81, 5]
sets2.corruption = False
sets2.layers = 1
sets2.patch_size = 5
sets2.strides = [1, 2, 2, 1]
sets2.padding = 'SAME'
sets2.depth_increasing = 4
sets2.residual_learning = RESIDUAL_LEARNING

sets3 = autoencoder.ConvAutoEncSettings()
sets3.prefix_name = "gamma"
sets3.input_shape = [BATCH_SIZE, 41, 41, 9]
sets3.corruption = False
sets3.layers = 1
sets3.patch_size = 5
sets3.strides = [1, 2, 2, 1]
sets3.padding = 'SAME'
sets3.depth_increasing = 4
sets3.residual_learning = RESIDUAL_LEARNING

sets4 = autoencoder.ConvAutoEncSettings()
sets4.prefix_name = "delta"
sets4.input_shape = [BATCH_SIZE, 21, 21, 13]
sets4.corruption = False
sets4.layers = 1
sets4.patch_size = 5
sets4.strides = [1, 2, 2, 1]
sets4.padding = 'SAME'
sets4.depth_increasing = 4
sets4.residual_learning = RESIDUAL_LEARNING

sets5 = autoencoder.ConvAutoEncSettings()
sets5.prefix_name = "epsilon"
sets5.input_shape = [BATCH_SIZE, 11, 11, 17]
sets5.corruption = False
sets5.layers = 1
sets5.patch_size = 5
sets5.strides = [1, 2, 2, 1]
sets5.padding = 'SAME'
sets5.depth_increasing = 4
sets5.residual_learning = RESIDUAL_LEARNING

sets6 = autoencoder.ConvAutoEncSettings()
sets6.prefix_name = "zeta"
sets6.input_shape = [BATCH_SIZE, 6, 6, 21]
sets6.corruption = False
sets6.layers = 1
sets6.patch_size = 5
sets6.strides = [1, 2, 2, 1]
sets6.padding = 'SAME'
sets6.depth_increasing = 4
sets6.residual_learning = RESIDUAL_LEARNING

sets = (
    sets1,
    sets2,
    sets3,
    sets4,
    sets5,
    sets6
)

# Do not edit below this line
#  ___         _  ___           __ _                    _   _
# | __|_ _  __| |/ __|___ _ _  / _(_)__ _ _  _ _ _ __ _| |_(_)___ _ _
# | _|| ' \/ _` | (__/ _ \ ' \|  _| / _` | || | '_/ _` |  _| / _ \ ' \
# |___|_||_\__,_|\___\___/_||_|_| |_\__, |\_,_|_| \__,_|\__|_\___/_||_|
#                                   |___/

with open(OUTPUT_FILE, "wb") as fp:
    pickle.dump(sets, fp, pickle.HIGHEST_PROTOCOL)

TEMPLATE_AE = """
AUTOENCODER STRUCTURE -----

  - blocks: {}
  - residual learning: {}
  - batch size: {}
"""

TEMPLATE_BLOCK = """
 [{} of {}] {}:
  - input shape: {} x {} x {}
  - corrupted: {}
  - layers: {}
  - filter: {} x {} with stride {} x {} and padding {}
"""

print(TEMPLATE_AE.format(len(sets),
  RESIDUAL_LEARNING,
  BATCH_SIZE))

for idx, block in enumerate(sets):
    print(TEMPLATE_BLOCK.format(idx + 1,
        len(sets),
        block.prefix_name,
        block.input_shape[1], block.input_shape[2], block.input_shape[3],
        block.corruption,
        block.layers,
        block.patch_size, block.patch_size, block.strides[1], block.strides[2], block.padding))

print("\n\nConfiguration written on: {}".format(OUTPUT_FILE))
