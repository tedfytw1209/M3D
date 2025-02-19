#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out


module load conda

conda activate m3d

python inference_ldctvqa.py --model_name_or_path GoodBaiBai88/M3D-LaMed-Llama-2-7B --image-folder /blue/chenaokun1990/tienyuchang/CT_nii/ --question-file Data/data/questions_VQA_large_nodule_test_nii_norm.jsonl --answers-file eval_output/answer_m3d_lnodule_vqa_0ep.jsonl

