#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=LOM2SEM-pix2pix-gpu-10k
#SBATCH --mail-user=namano@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=40:00:00
#SBATCH --account=eaholm0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=/home/%u/logs/LOM2SEM/%x-%j.log

# The application(s) to execute along with its input arguments and options:

cd ~/research/LOM2SEM
module load python/3.12.1
source venv/L2S/bin/activate
python -m Pix2Pix.train_p2phd --config Pix2Pix/configs/p2phd_aug_gpu_long.yaml
