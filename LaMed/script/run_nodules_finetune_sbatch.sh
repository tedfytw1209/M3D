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

bash LaMed/script/finetune_lora_llama2_ldct_nodules.sh
