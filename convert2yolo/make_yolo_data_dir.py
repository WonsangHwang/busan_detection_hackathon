"""
대회에 제공된 data를 YOLO에 입력할 수 있는 형태로 변환한다.
아래와 같은 디렉토리 구조를 생성한 후,
<dst dir>
        ├── images
                ├── train
                ├── val
                └── test
        └── labels
                ├── train
                ├── val
                └── test
이미지 파일 및 annotation 파일을 해당되는 디렉토리에 저장한다.

train/val/test 구분은 입력되는 json 파일에 의해 결정된다.
json 파일은 split_data.py를 통해 생성할 수 있다.

annotation 파일은 대응되는 이미지 파일과 확장자를 제외하고는 이름이 같은 txt 파일로 저장한다.
각 line은 하나의 bbox를 의미 하며,
label, box_center_x, box_center_y, box_width, box_height 정보가 space로 구분되어 write되어 있어야 한다.

대회에서 제공되는 data는 주된 object로 구별되어 디렉토리 별로 나누어져 있으며,
하나의 디렉토리 내에는 이미지 파일과 이미지 메타 정보와, bbox의 정보를 갖는 json 파일 들이 있다.
이미지 파일과 json의 이름은 확장자를 제외하고 같다.
"""
import argparse
import os
import json
import shutil
from typing import Dict, List, Tuple
from tqdm import tqdm

# 대회에서 사용하는 class id 와 class name
obj_id_to_class_name_dict = {
    9: 'garbage_bag',
    13: 'banner',
    14: 'tent',
    17: 'pet',
    18: 'fence',
    19: 'bench',
    20: 'park_pot',
    21: 'trash_can',
    22: 'rest_area',
    23: 'toilet',
    24: 'park_headstone',
    25: 'street_lamp',
    26: 'park_info'
}

obj_id_to_label: Dict[int, int] = {obj_id: label for label, obj_id in enumerate(obj_id_to_class_name_dict.keys())}

# class id를 YOLO에서 그대로 사용할 수 없으므로, 순서대로 0 ~ 12의 번호로 label을 설정한다.
obj_label_to_id: Dict[int, int] = {obj_label: obj_id for obj_id, obj_label in obj_id_to_label.items()}


def get_input_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='json file including image file paths for train/val')
    parser.add_argument('--test', type=str, help='json file including image file paths for test')
    parser.add_argument('--src', type=str, help='original data\'s dir')
    parser.add_argument('--dst', type=str, help='destination dir to save / yolo format data dir')

    return parser.parse_args()


def make_yolo_dir_tree(dst_dir: str) -> Tuple[str, str, str, str, str, str, str, str]:
    """
    yolo format에 맞는 아래과 같은 dir tree를 만든다.
    같은 이름의 디렉토리가 존재하면 삭제하고, 다시 생성한다.
    <dst dir>
        ├── images
                ├── train
                ├── val
                └── test
        └── labels
                ├── train
                ├── val
                └── test
    Args:
        dst_dir:

    Returns: 생성되는 되는 8개의 디렉토리 경로를 순서대로 반환한다.
    """
    dst_images_dir = os.path.join(dst_dir, 'images')
    dst_images_train_dir = os.path.join(dst_images_dir, 'train')
    dst_images_val_dir = os.path.join(dst_images_dir, 'val')
    dst_images_test_dir = os.path.join(dst_images_dir, 'test')

    dst_labels_dir = os.path.join(dst_dir, 'labels')
    dst_labels_train_dir = os.path.join(dst_labels_dir, 'train')
    dst_labels_val_dir = os.path.join(dst_labels_dir, 'val')
    dst_labels_test_dir = os.path.join(dst_labels_dir, 'test')

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)

    os.mkdir(dst_images_dir)
    os.mkdir(dst_images_train_dir)
    os.mkdir(dst_images_val_dir)
    os.mkdir(dst_images_test_dir)

    os.mkdir(dst_labels_dir)
    os.mkdir(dst_labels_train_dir)
    os.mkdir(dst_labels_val_dir)
    os.mkdir(dst_labels_test_dir)

    return dst_images_dir, dst_images_train_dir, dst_images_val_dir, dst_images_test_dir, \
        dst_labels_dir, dst_labels_train_dir, dst_labels_val_dir, dst_labels_test_dir


def get_yolo_bboxes(ann_json_file_path: str) -> Tuple[List[Tuple[int, float, float, float, float]], bool]:
    """
    한 이미지 내 모든 bbox 정보를 yolo format 으로 변환한다.
    make_yolo_ann_txt_file 에 의해 호출되어 사용된다.
    Args:
        ann_json_file_path: 한 이미지의 bbox 정보를 담은 json (대회 제공) 경로 - 소스 root dir 까지 포함한 경로

    Returns:
        list of yolo format bbox info (label, box_center_x, box_center_y, box_width, box_height)
        All coordinates are normalized to [0, 1]
        and
        whether the ann json file have wrong class ids
    """
    with open(ann_json_file_path, 'r') as ann_file:
        ann_json_data = json.load(ann_file)
        width = ann_json_data['images']['width']
        height = ann_json_data['images']['height']

        yolo_bboxes = list()
        key_error_yn = False
        for pixel_bbox in ann_json_data['annotations']:
            try:
                label = obj_id_to_label[pixel_bbox['object_id']]
            except KeyError:
                key_error_yn = True
                continue
            box_center_x = (pixel_bbox['bbox'][1][0] + pixel_bbox['bbox'][0][0]) / 2 / width
            box_center_y = (pixel_bbox['bbox'][1][1] + pixel_bbox['bbox'][0][1]) / 2 / height
            box_width = (pixel_bbox['bbox'][1][0] - pixel_bbox['bbox'][0][0]) / width
            box_height = (pixel_bbox['bbox'][1][1] - pixel_bbox['bbox'][0][1]) / height
            yolo_bboxes.append(
                (label, box_center_x, box_center_y, box_width, box_height)
            )

        return yolo_bboxes, key_error_yn


def make_yolo_ann_txt_file(ann_json_file_path: str, dst_dir: str) -> bool:
    """
    하나의 이미지에 대한 annotation json을 load 하여, yolo형식의 annotation txt 파일을 생성한다.
    Args:
        ann_json_file_path: 한 이미지의 bbox 정보를 담은 json (대회 제공) 경로 - 소스 root dir 까지 포함한 경로
        dst_dir: ann txt 를 저장할 경로

    Returns: whether the ann json file have wrong class ids
    """
    json_file_name = os.path.basename(ann_json_file_path)
    txt_file_name = json_file_name.replace(".json", ".txt")
    txt_file_path = os.path.join(dst_dir, txt_file_name)

    yolo_bboxes: List[Tuple[int, float, float, float, float]]
    key_error_yn: bool
    yolo_bboxes, key_error_yn= get_yolo_bboxes(ann_json_file_path)
    with open(txt_file_path, 'w') as txt_file:
        for bbox in yolo_bboxes:
            txt_file.write('%d %f %f %f %f\n' % bbox)

    return key_error_yn


def make_yolo_ann_txt_files(src_root_dir: dir, src_img_file_list: List[str], dst_dir: str) -> None:
    """
    입력된 이미지 파일 리스트의 각 이미지에 대한
    annotation json을 load 하여, yolo형식의 annotation txt 파일을 생성한다.
    Args:
        src_root_dir: 대회 제공 원본데이터가 저장된 디렉토리
        src_img_file_list: 이미지 파일 리스트, src_root_dir 에서의 상대 경로
        dst_dir: ann txt 를 저장할 경로

    Returns: None
    """
    print("make_yolo_ann_txt_files")
    key_error_files = list()
    for src_img_file in tqdm(src_img_file_list):
        src_img_file_path = os.path.join(src_root_dir, src_img_file)
        ann_json_file_path = src_img_file_path.replace(".jpg", ".json")
        key_error_yn = make_yolo_ann_txt_file(ann_json_file_path, dst_dir)

        if key_error_yn:
            key_error_files.append(ann_json_file_path)

    print("The following files have one or more bboxes with a wrong class id")
    print(key_error_files)


def copy_files(src_root_dir: str, src_file_list: List[str], dst_dir: str) -> None:
    """
    파일 들을 원하는 경로로 복사한다.
    Args:
        src_root_dir: 대회 제공 원본데이터가 저장된 디렉토리
        src_file_list: 복사할 이미지 파일 리스트, src_root_dir 에서의 상대 경로
        dst_dir: 복사된 데이터가 저장되는 경로

    Returns: None
    """
    print("copy files")
    for src_file in tqdm(src_file_list):
        src_file_path = os.path.join(src_root_dir, src_file)
        dst_file_path = os.path.join(dst_dir, os.path.basename(src_file))
        shutil.copyfile(src_file_path, dst_file_path)


if __name__ == '__main__':
    in_args = get_input_args()

    src_data_root_dir = in_args.src

    dst_img_dir, dst_img_train_dir, dst_img_val_dir, dst_img_test_dir, \
        dst_label_dir, dst_label_train_dir, dst_label_val_dir, dst_label_test_dir = make_yolo_dir_tree(in_args.dst)

    print("====== Train ===========================================================================")
    with open(in_args.train) as json_file:
        file_list = json.load(json_file)['train']
        copy_files(src_data_root_dir, file_list, dst_img_train_dir)
        make_yolo_ann_txt_files(src_data_root_dir, file_list, dst_label_train_dir)

    print("====== Validation ======================================================================")
    with open(in_args.train) as json_file:
        file_list = json.load(json_file)['val']
        copy_files(src_data_root_dir, file_list, dst_img_val_dir)
        make_yolo_ann_txt_files(src_data_root_dir, file_list, dst_label_val_dir)

    print("====== Test ============================================================================")
    with open(in_args.test) as json_file:
        file_list = json.load(json_file)
        copy_files(src_data_root_dir, file_list, dst_img_test_dir)
        make_yolo_ann_txt_files(src_data_root_dir, file_list, dst_label_test_dir)


