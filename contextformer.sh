#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=24:00:00
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --dependency=afterany:123371

python /home/nikoskot/greenearthnet/train.py /home/nikoskot/greenearthnet/model_configs/contextformer/contextformer6M/seed=27.yaml --data_dir /home/nikoskot/earthnetThesis/GreenEarthnetDataset/greenearthnet/