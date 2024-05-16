#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=LOM2SEM-style-gpu
#SBATCH --mail-user=namano@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=01:00:00
#SBATCH --account=eaholm0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/logs/LOM2SEM/stylegan/%x-%j.log

# The application(s) to execute along with its input arguments and options:

cd ~/research/LOM2SEM
source venv/L2S/bin/activate
python ./stylegan2/train.py --outdir=/nfs/turbo/coe-eaholm/namano/LOM2SEM/results/stylegan2 --lomdata=/nfs/turbo/coe-eaholm/namano/LOM2SEM/mecs_steel/train_LOM512/ --semdata=/nfs/turbo/coe-eaholm/namano/LOM2SEM/mecs_steel/train_SEM512/
