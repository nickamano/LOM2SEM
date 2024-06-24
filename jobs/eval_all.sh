#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=LOM2SEM-eval-all
#SBATCH --mail-user=namano@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2g
#SBATCH --time=1:00:00
#SBATCH --account=eaholm0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/logs/LOM2SEM/eval/all/%x-%j.log

# The application(s) to execute along with its input arguments and options:

cd ~/research/LOM2SEM
module load python/3.12.1
source venv/L2S/bin/activate
# python eval.py \
#                     -t /nfs/turbo/coe-eaholm/namano/LOM2SEM/mecs_steel/test_SEM512_forFID \
#                     -g /nfs/turbo/coe-eaholm/namano/LOM2SEM/results/FID/pix2pix/p2phd_aug_gpu_test_FID_20240616_104405/gen/test \
#                     -d /nfs/turbo/coe-eaholm/namano/LOM2SEM/results/eval/all/pix2pix/FID
# python eval.py \
#                     -t /nfs/turbo/coe-eaholm/namano/LOM2SEM/mecs_steel/test_SEM512_forFID \
#                     -g /nfs/turbo/coe-eaholm/namano/LOM2SEM/results/FID/AdaIN/AdaIn_gpu_test_FID_20240616_104205/gen/test \
#                     -d /nfs/turbo/coe-eaholm/namano/LOM2SEM/results/eval/all/adain/FID
python eval.py \
                    -t /nfs/turbo/coe-eaholm/namano/LOM2SEM/mecs_steel/test_SEM512_forFID \
                    -g /nfs/turbo/coe-eaholm/namano/LOM2SEM/results/FID/Palette/test_gd_xavier_in_all_240617_072327/results/test/0/Out \
                    -d /nfs/turbo/coe-eaholm/namano/LOM2SEM/results/eval/all/palette/FID
