#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=LOM2SEM-eval-palette-gd
#SBATCH --mail-user=namano@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=1:00:00
#SBATCH --account=eaholm0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/logs/LOM2SEM/eval/palette/sr3/%x-%j.log

# The application(s) to execute along with its input arguments and options:

cd ~/research/LOM2SEM
module load python/3.12.1
source venv/L2S/bin/activate
python eval.py \
                    -t /nfs/turbo/coe-eaholm/namano/LOM2SEM/mecs_steel/test_SEM512 \
                    -g /nfs/turbo/coe-eaholm/namano/LOM2SEM/results/Palette/test_sr3_kaiming_in_all_240613_093731/results/test/0/Out/good_seed \
                    -d /nfs/turbo/coe-eaholm/namano/LOM2SEM/results/eval/palette/
