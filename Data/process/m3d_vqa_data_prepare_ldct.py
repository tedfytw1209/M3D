import os
from tqdm import tqdm
from examples.vllm_wrapper import vLLMWrapper
import re
import csv
import json

# multiple groups in parallel improves speed
group_id = 0
group_num = 4

root_path = 'PATH/data/M3D_Cap_npy/'
file_path = "PATH/data/3DCTTEXT_npy/M3D_Cap.json"
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)


train_data = data.get('train', [])
total_items = len(train_data)
chunk_size = total_items // group_num

split_train_data = [train_data[i:i + chunk_size] for i in range(0, total_items, chunk_size)]

data_list = split_train_data[group_id]
data_len = len(data_list)
print("data_len: ",data_len)

vqa_data_name = "M3D_VQA_" + str(group_id) + ".csv"
path = "PATH/Qwen/VQA-data/" + vqa_data_name
with open(path, "a", newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        ["Image Path", "Text", "Question Type", "Question", "Choice A", "Choice B", "Choice C", "Choice D", "Answer", "Answer Choice"])
'''
M3D_Cap_npy/ct_case/007789/Axial_Zoomed___thin_cuts.npy
Tree in bud nodules noted within the left inferior lingula with peribronchial thickening, may suggest pulmonary infection. Few small subcentimeter reactive mediastinal lymph nodes.Subglottic endotracheal polyps are noted averaging 3 mm, likely fibroepithelial polyps.Tiny posterior tracheal diverticulae are noted.Scans through the upper abdomen revealed enlarged fatty liver (20 cm in MCL).
3
Which organ is primarily affected by the abnormality described in the image?
Lung
Liver
Trachea
Lymph nodes
Lung
A
'''
for i, data in tqdm(enumerate(data_list)):
    try:
        image_file = os.path.basename(data["image"])

        with open(data["text"], "r") as f:
            text = f.read()

        with open(path, "a", newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            question_num, question, choices, an_choice, answer = q
            choices = re.findall(r"([A-D])\. (.+?)(?=(?: [A-D]\.|$))", choices)
            choices_dict = {choice[0]: choice[1] for choice in choices}

            for option in ['A', 'B', 'C', 'D']:
                if option not in choices_dict:
                    choices_dict[option] = 'NA'

            if int(question_num) < 4:
                question_type = question_num
            elif int(question_num) < 7:
                question_type = str(4)
            else:
                question_type = str(5)

            csvwriter.writerow(
                [data["image"], text, question_type, question, choices_dict['A'], choices_dict['B'], choices_dict['C'], choices_dict['D'],
                    answer, an_choice])
    except:
        print("Error in " + "id:" + str(i) + " " + data["image"])
