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

@dataclass
class AllArguments:
    model_name_or_path: str = field(default="GoodBaiBai88/M3D-LaMed-Llama-2-7B")

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
    args = parser.parse_args_into_dataclasses()[0]

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

    question = report_gen_prompts[0]
    # question = "What is liver in this image? Please output the segmentation mask."
    # question = "What is liver in this image? Please output the box."

    image_tokens = "<im_patch>" * args.proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

    if args.image_path.endswith('.nii') or args.image_path.endswith('.nii.gz'):
        image_np = nib.load(args.image_path).get_fdata()
        image_np = nii_process(image_np)
    else:
        image_np = np.load(args.image_path)
    print(image_np.shape)
    image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

    generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
    # generation, seg_logit = model.generate(image_pt, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

    generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
    #seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0

    print('question', question)
    print('generated_texts', generated_texts[0])

    # image = image_np[0]
    # slice = image.shape[0]
    # for i in range(slice):
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image[i], cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(seg_mask[0][0][i].cpu().numpy(), cmap='gray')
    #     plt.axis('off')
    #     plt.show()

    '''image = sikt.GetImageFromArray(image_np)
    ssv.display(image)

    seg = sikt.GetImageFromArray(seg_mask.cpu().numpy()[0])
    ssv.display(seg)'''

if __name__ == "__main__":
    main()
