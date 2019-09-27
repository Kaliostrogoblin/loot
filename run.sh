#!/bin/sh
module add Python/v3.6.5
module add GVR/v1.0-1
pip install -r requirements.txt --user
sbatch slurm_script.sh