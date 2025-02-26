#!/bin/bash

module load conda 
source activate overcooked_ai

tensorboard --logdir=runs --host "0.0.0.0" --port 6008

echo 'sudo ssh -L 6007:localhost:6008 sylve057@agate.msi.umn.edu'
echo 'Access TensorBoard at http://localhost:6008'

