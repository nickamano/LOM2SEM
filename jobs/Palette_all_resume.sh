#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=LOM2SEM-pal-all-resume
#SBATCH --mail-user=namano@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=17:00:00
#SBATCH --account=eaholm0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=/home/%u/logs/LOM2SEM/palette/all/%x-%j.log

# The application(s) to execute along with its input arguments and options:

cd ~/research/LOM2SEM
module load python/3.12.1
source venv/L2S/bin/activate
# python -m Palette.run -p train -c Palette/config/resume_gd_kaiming.json 
# python -m Palette.run -p train -c Palette/config/resume_gd_normal.json 
# python -m Palette.run -p train -c Palette/config/resume_gd_xavier.json 
# python -m Palette.run -p train -c Palette/config/resume_sr3_kaiming.json 
python -m Palette.run -p train -c Palette/config/resume_sr3_normal.json 
python -m Palette.run -p train -c Palette/config/resume_sr3_xavier.json 
