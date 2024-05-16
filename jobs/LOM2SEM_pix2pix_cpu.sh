#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=gl-LOM2SEM1
#SBATCH --mail-user=namano@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=08:00:00
#SBATCH --account=eaholm0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/logs/LOM2SEM/%x-%j.log

# The application(s) to execute along with its input arguments and options:

cd ~/research/LOM2SEM
source venv/L2S/bin/activate
python -m Pix2Pix.train_p2phd --config Pix2Pix/configs/p2phd_aug.yaml
