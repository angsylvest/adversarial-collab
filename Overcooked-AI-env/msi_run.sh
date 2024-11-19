#!/bin/bash -l        
#SBATCH --time=48:00:00
#SBATCH --ntasks=12
#SBATCH --mem=70g
#SBATCH --tmp=70g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=sylve057@umn.edu
module load conda 
source activate overcooked_ai

python ~/adversarial-collab/Overcooked-AI/train_online.py agent=maddpg save_replay_buffer=true


