#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out


module load conda

conda activate m3d

python inference.py --model_name_or_path /orange/chenaokun1990/tienyu/m3d_model/m3d_ldct_lora_2ep --image-folder /blue/chenaokun1990/tienyuchang/CT_nii/ \
 --question-file Data/data/questions_Report_notes_test_nii.jsonl --answers-file eval_output/answer_m3d_lora_notes.jsonl

#python inference.py --model_name_or_path /orange/chenaokun1990/tienyu/m3d_model/m3d_ldct_lora_2ep --image-folder /blue/chenaokun1990/tienyuchang/CT_nii/ \
# --question-file Data/data/questions_Report_nodules_test_nii.jsonl --answers-file eval_output/answer_m3d_lora_notes.jsonl
