import os
import numpy as np
import pandas as pd
import nibabel as nib
import json
from PIL import Image
import concurrent.futures
from tqdm import tqdm
from collections import Counter
import unicodedata
import monai.transforms as mtf
from multiprocessing import Pool
from unidecode import unidecode

# input_dir = 'PATH/M3D_Cap/ct_quizze/'
# output_dir = 'PATH/M3D_Cap_npy/ct_quizze/'

input_dir = '/blue/bianjiang/tienyuchang/CT_nii/'
output_dir = '/blue/bianjiang/tienyuchang/CT_npy/'

# Get all subfolders [00001, 00002....]
#subfolders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]


transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
])

def load_jsonl_file(jsonl_file):
    try:
        with open(jsonl_file) as fp:
            list_data_dict = json.load(fp)
    except:
        with open(jsonl_file) as fp:
            list_data_dict = [json.loads(q) for q in fp]
    return list_data_dict

def process_nii_files(nii_file):
    output_id_folder = output_dir
    input_id_folder = input_dir
    nii_name = nii_file.replace('.nii.gz', '')

    os.makedirs(output_id_folder, exist_ok=True)

    output_path = os.path.join(output_dir, f'{nii_name}.npy')

    try:
        final_3d_image = nib.load(os.path.join(input_id_folder,nii_file)).get_fdata()
        image = final_3d_image[np.newaxis, ...]
        image = image - image.min()
        image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
        img_trans = transform(image)
        np.save(output_path, img_trans)
    except:
        print(final_3d_image.shape)
        print("This folder is vstack error: ", output_path)


data_path = "/blue/bianjiang/tienyuchang/VILA/playground/data/eval/LungCancer_3DCT/questions_Report_notes_train_nii.jsonl"
data_path2 = "/blue/bianjiang/tienyuchang/VILA/playground/data/eval/LungCancer_3DCT/questions_Report_notes_test_nii.jsonl"
list_data_dict = load_jsonl_file(data_path) + load_jsonl_file(data_path2)
nii_files = [data_dict['nii'] for data_dict in list_data_dict]

with Pool(processes=32) as pool:
    with tqdm(total=len(nii_files), desc="Processing") as pbar:
        for _ in pool.imap_unordered(process_nii_files, nii_files):
            pbar.update(1)
