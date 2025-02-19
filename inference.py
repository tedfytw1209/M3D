import argparse
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import nibabel as nib
import torch
from dataclasses import dataclass, field
import simple_slice_viewer as ssv
import SimpleITK as sikt
# from LaMed.src.model.language_model import *
import matplotlib.pyplot as plt
import monai.transforms as mtf
import json
from tqdm import tqdm
import shortuuid

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
])
def nii_process(nii_arr):
    """_summary_

    Args:
        nii_arr (np.array): input np.array of nii image (H,W,D)

    Returns:
        img_trans (np.array): output np.array of nii image (D,H,W)
    """
    image = np.transpose(nii_arr,(2,0,1))[np.newaxis, ...]
    #value min/max
    image = image - image.min()
    image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
    #transform
    img_trans = transform(image)
    return img_trans
#get img key
def get_img_key(question_keys):
    if 'image' in question_keys:
        img_key = 'image'
    elif 'filename' in question_keys:
        img_key = 'filename'
    elif "image:" in question_keys:
        img_key = 'image:'
    else:
        img_key = "image"
    return img_key
##data save
def save_jsonl(out_path,data):
    with open(out_path, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')

@dataclass
class AllArguments:
    model_name_or_path: str = field(default="/orange/chenaokun1990/tienyu/m3d_model/m3d_ldct_lora_2ep")

    proj_out_num: int = field(default=256, metadata={"help": "Number of output tokens in Projector."})
    image_path: str = field(default="./Data/data/examples/example_04.npy")

report_gen_prompts = [
    "Can you provide a caption consists of findings for this medical image?",
    "Describe the findings of the medical image you see.",
    "Please caption this medical scan with findings.",
    "What is the findings of this image?",
    "Describe this medical scan with findings.",
    "Please write a caption consists of findings for this image.",
    "Can you summarize with findings the images presented?",
    "Please caption this scan with findings.",
    "Please provide a caption consists of findings for this medical image.",
    "Can you provide a summary consists of findings of this radiograph?",
    "What are the findings presented in this medical scan?",
    "Please write a caption consists of findings for this scan.",
    "Can you provide a description consists of findings of this medical scan?",
    "Please caption this medical scan with findings.",
    "Can you provide a caption consists of findings for this medical scan?"
]

def main():
    seed_everything(42)
    device = torch.device('cuda') # 'cpu', 'cuda'
    dtype = torch.float16 # or bfloat16, float16, float32

    parser = transformers.HfArgumentParser(AllArguments)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    args = parser.parse_args_into_dataclasses()[0]
    add_args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map='cuda',
        trust_remote_code=True
    )
    model = model.to(device=device)
    
    #inference data
    questions = [json.loads(q) for q in open(os.path.expanduser(add_args.question_file), "r")]
    question_keys = [k for k in questions[0].keys()]
    img_key = get_img_key(question_keys)
    ques_key = 'text'
    out_data = []
    ans_file = open(add_args.answers_file, "w")
    i = 0
    for question_row in tqdm(questions):
        i += 1
        if 'question_id' in question_keys:
            idx = question_row["question_id"]
        elif 'image_id' in question_keys:
            idx = question_row["image_id"]
        else:
            idx = i
        question = report_gen_prompts[0]
        # question = "What is liver in this image? Please output the segmentation mask."
        # question = "What is liver in this image? Please output the box."

        image_tokens = "<im_patch>" * args.proj_out_num
        input_txt = image_tokens + question
        input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
        img_path = os.path.join(add_args.image_folder, question_row[img_key])
        try:
            if img_path.endswith('.nii') or img_path.endswith('.nii.gz'):
                image_np = nib.load(img_path).get_fdata()
                image_np = nii_process(image_np)
                image_pt = torch.Tensor(image_np).unsqueeze(0).to(dtype=dtype, device=device)
            else:
                image_np = np.load(img_path)
                image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
            #print(image_np.shape)

            generation = model.generate(image_pt, input_id, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=1.0)
            # generation, seg_logit = model.generate(image_pt, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
        except Exception as e:
            print(f"Error: {e}")
            generated_texts = ["Error: " + str(e)]
        #seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
        #print('question', question)
        #print('generated_texts', generated_texts[0])
        ans_id = shortuuid.uuid()
        out_dict = {"question_id": idx,
                    "prompt": question,
                    "text": generated_texts[0],
                    "answer_id": ans_id,
                    "model_id": args.model_name_or_path,
                    "metadata": {}}
        ans_file.write(json.dumps(out_dict) + "\n")
        ans_file.flush()
        out_data.append(out_dict)
    ans_file.close()
    #save_jsonl(file_args.answers_file, out_data)


if __name__ == "__main__":
    main()
