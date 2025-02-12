#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=yonghui.wu
#SBATCH --qos=yonghui.wu
#SBATCH --reservation=gatortrongpt


module load conda

conda activate m3d

python inference.py --model_name_or_path GoodBaiBai88/M3D-LaMed-Llama-2-7B --image-folder /blue/bianjiang/tienyuchang/CT_nii/ --question-file Data/data/questions_Report_notes_test_nii.jsonl --answers-file eval_output/answer_m3d_notes.jsonl

python inference.py --model_name_or_path GoodBaiBai88/M3D-LaMed-Llama-2-7B --image-folder /blue/bianjiang/tienyuchang/CT_nii/ --question-file Data/data/questions_Report_nodules_test_nii.jsonl --answers-file eval_output/answer_m3d_nodules.jsonl

