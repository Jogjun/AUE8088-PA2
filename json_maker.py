import json
import os
from tqdm import tqdm
from glob import glob

def parse_txt_files(txt_file_paths):
    images = []
    annotations = []
    categories = [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "cyclist"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "person?"}
    ]
    
    image_id = 0
    annotation_id = 0

    for txt_file_path in txt_file_paths:
        txt_file_path = txt_file_path.strip()
        # This is an image name
        im_name = "/".join(txt_file_path.split(os.sep)[-1][:-4].split("_"))
        
        images.append({
            "id": image_id,
            "im_name": im_name,
            "height": 512,  # assuming fixed height
            "width": 640   # assuming fixed width
        })
        
        gt_txt_path = os.path.join('/'.join(txt_file_path.split('/')[:-3]), 'labels', txt_file_path.split(os.sep)[-1][:-4] + '.txt')
        with open(gt_txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                # This is annotation data
                category_id = int(parts[0])
                x_center = int(float(parts[1]) * 640)
                y_center = int(float(parts[2]) * 512)
                width = int(float(parts[3]) * 640)
                height = int(float(parts[4]) * 512)
                x = x_center + width
                y = y_center + height
                bbox = [x, y, width, height]
                occlusion = int(parts[5])
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "height": height,
                    "occlusion": occlusion,
                    "ignore": 0  # assuming no ignore flag
                })
                annotation_id += 1
    
        image_id += 1        
    return {
        "info": {
            "dataset": "KAIST Multispectral Pedestrian Benchmark",
            "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
            "related_project_url": "http://multispectral.kaist.ac.kr",
            "publish": "CVPR 2015"
        },
        "info_improved": {
            "sanitized_annotation": {
                "publish": "BMVC 2018",
                "url": "https://li-chengyang.github.io/home/MSDS-RCNN/",
                "target": "files in train-all-02.txt (set00-set05)"
            },
            "improved_annotation": {
                "url": "https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn",
                "publish": "BMVC 2016",
                "target": "files in test-all-20.txt (set06-set11)"
            }
        },
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

def save_json(data, output_file_path):
    with open(output_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def save_txt(data_path, output_txt_path):
    # Save the data in txt format
    with open(output_txt_path, 'w') as file:
        for line in data_path:
            file_name = line.split(os.sep)[-1]
            file_name = file_name.replace('.txt', '.jpg')
            prefix = "datasets/kaist-rgbt/train/images/{}"
            dst = os.path.join(prefix, file_name)
            file.write(dst + '\n')
    
if __name__ == "__main__":
    # txt_file_path = glob('/home/ailab/AILabDataset/03_Shared_Repository/jonghyun/Study/kaist-rgbt/test/images/visible/*.jpg')

    # # read txt
    # with open('/home/ailab/AILabDataset/03_Shared_Repository/jonghyun/Study/AUE8088-PA2/datasets/kaist-rgbt/test-all-20.txt', 'r') as file:
    #     txt_file_path = file.readlines()
    # data = parse_txt_files(txt_file_path)
    # output_json_path = './debug/KAIST_annotation.json'
    # save_json(data, output_json_path)        

    with open('/home/hyu3/Choi/aue8088-pa2/datasets/kaist-rgbt/val_sets.txt', 'r') as file:
        txt_file_path = file.readlines()                
    
    # for i in range(5):
    #     train_file_path = [p for p in txt_file_path if 'set{:02d}'.format(i) not in p]
    #     val_file_path  = [p for p in txt_file_path if 'set{:02d}'.format(i) in p]
            
    #     # make dataset for k-fold cross validation
    #     data = parse_txt_files(train_file_path)
    #     output_json_path = './debug/train_{:02d}_fold.json'.format(i)
    #     output_txt_path = './debug/train_{:02d}_fold.txt'.format(i)
    #     save_json(data, output_json_path)
    #     save_txt(train_file_path, output_txt_path)
        
    #     # validation dataset
    #     data = parse_txt_files(val_file_path)
    #     output_json_path = './debug/valid_{:02d}_fold.json'.format(i)
    #     output_txt_path = './debug/valid_{:02d}_fold.txt'.format(i)
    #     save_json(data, output_json_path)
    #     save_txt(val_file_path, output_txt_path)
    
    data = parse_txt_files(txt_file_path)
    output_json_path = './utils/eval/KAIST_annotation.json'
    save_json(data, output_json_path)  
        