import json
import os
from tqdm import tqdm
from glob import glob

def parse_annotation_files(annotation_file_paths):
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

    for annotation_file_path in annotation_file_paths:
        annotation_file_path = annotation_file_path.strip()
        # 이미지 이름 처리
        image_name = "/".join(annotation_file_path.split(os.sep)[-1][:-4].split("_"))
        
        images.append({
            "id": image_id,
            "file_name": image_name,
            "height": 512,  # 고정 높이 가정
            "width": 640   # 고정 너비 가정
        })
        
        # 레이블 파일 경로 생성
        label_file_path = os.path.join('/'.join(annotation_file_path.split('/')[:-3]), 'labels', annotation_file_path.split(os.sep)[-1][:-4] + '.txt')
        with open(label_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                # 어노테이션 데이터
                category_id = int(parts[0])
                x_center = int(float(parts[1]) * 640)
                y_center = int(float(parts[2]) * 512)
                width = int(float(parts[3]) * 640)
                height = int(float(parts[4]) * 512)
                x = x_center - (width / 2)
                y = y_center - (height / 2)
                bbox = [x, y, width, height]
                occlusion = int(parts[5])
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "height": height,
                    "occlusion": occlusion,
                    "ignore": 0  # ignore 플래그 없음 가정
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

def save_image_paths(image_paths, output_txt_path):
    # txt 형식으로 데이터 저장
    with open(output_txt_path, 'w') as file:
        for image_path in image_paths:
            file_name = image_path.split(os.sep)[-1]
            file_name = file_name.replace('.txt', '.jpg')
            prefix = "datasets/kaist-rgbt/train/images/{}"
            dst = os.path.join(prefix, file_name)
            file.write(dst + '\n')
    
if __name__ == "__main__":
    # 파일 읽기
    with open('/home/hyu3/Choi/aue8088-pa2/datasets/kaist-rgbt/train-all-04.txt', 'r') as file:
        annotation_file_paths = file.readlines()                
    
    for i in range(5):
        train_file_paths = [p for p in annotation_file_paths if 'set{:02d}'.format(i) not in p]
        val_file_paths  = [p for p in annotation_file_paths if 'set{:02d}'.format(i) in p]
            
        # k-fold 교차 검증용 데이터셋 생성
        train_data = parse_annotation_files(train_file_paths)
        output_train_json_path = './debug/train_{:02d}_fold.json'.format(i)
        output_train_txt_path = './debug/train_{:02d}_fold.txt'.format(i)
        save_json(train_data, output_train_json_path)
        save_image_paths(train_file_paths, output_train_txt_path)
        
        # 검증 데이터셋
        val_data = parse_annotation_files(val_file_paths)
        output_val_json_path = './debug/valid_{:02d}_fold.json'.format(i)
        output_val_txt_path = './debug/valid_{:02d}_fold.txt'.format(i)
        save_json(val_data, output_val_json_path)
        save_image_paths(val_file_paths, output_val_txt_path)
