#!/bin/bash

#SBATCH --partition=A100-40GB,A100-80GB,H100,H200
#SBATCH --gpus=1                          
#SBATCH --mem=64G                         
#SBATCH --cpus-per-task=32                
#SBATCH --time=23:00:00                    
#SBATCH --output=./logs/slurm_outputs/slurm-%j.out   
#SBATCH --error=./logs/slurm_outputs/slurm-%j.err    
#SBATCH --job-name=ik_net-estimator-train-%j               

export PROJECT_ROOT="/cmillerd/experiments/Aria/ik_net_develop/slurm_scripts"
COMMAND="cd ${PROJECT_ROOT}; python train.py"


srun \
    --container-image=/netscratch/asodariya/ubuntu+20.04_v2.sqsh \
    --container-mounts=/fscratch/$USER:/fscratch/$USER,/netscratch:/netscratch,/home/$USER/:/cmillerd,/ds:/ds,/ds-av:/ds-av  \
    --task-prolog=/cmillerd/run.sh \
    bash -ic "set -m && source /cmillerd/.bashrc && $COMMAND"

