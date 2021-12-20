"""
대회 제공 데이터를 train / val / test 로 나눈다.
train, val, test 쓰여질 파일 리스트를 json으로 저장한다.

입력되는 test-ratio 비율 만큼의 데이터가 test용 으로 할당된다.
train/val 은 k-fold 방식으로 나누며, k개의 set이 생성된다.
"""
import argparse
import json
import os
import random
from typing import Dict, List

import numpy as np


def get_input_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='original data\'s dir')
    parser.add_argument('--dst', type=str, default='./', help='destination dir to save json files in')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='test data ratio in entire data')
    parser.add_argument('--k', type=int, default=9, help='k value in \'k-fold\'')
    parser.add_argument('--seed', type=int, default=None, help='random seed, used when shuffling data')

    return parser.parse_args()


if __name__ == '__main__':
    in_args = get_input_args()

    if in_args.seed:
        random.seed(in_args.seed)

    src_data_root_dir = in_args.src
    dst_dir = in_args.dst
    k = in_args.k

    # 하위 디렉토리 보기 - 절대 경로
    sub_dirs = [os.path.join(src_data_root_dir, dir_name) for dir_name in os.listdir(src_data_root_dir)]
    print('Sub directories')
    print(sub_dirs)

    # 하위 디렉토리 별로 이미지들을 train/val/test로 나눈 뒤 통합한다.
    # 입력된 test ratio 대로 test data가 추출된 후,
    # 나머지 이미지들에 대해 k개의 train/val set 을 만든다.
    test_img_file_paths: List[str] = list()
    train_val_img_file_paths_k_sets: List[Dict[str, List[str]]] = [{'train': [], 'val': []} for _ in range(k)]

    for sub_dir in sub_dirs:
        img_file_paths = [os.path.join(sub_dir, file_name) for file_name in
                          os.listdir(os.path.join(src_data_root_dir, sub_dir)) if file_name.endswith('.jpg')]
        random.shuffle(img_file_paths)

        test_img_file_paths.extend(img_file_paths[:int(len(img_file_paths) * in_args.test_ratio)])
        train_val_img_file_paths = img_file_paths[int(len(img_file_paths) * in_args.test_ratio):]

        k_splits = np.array_split(train_val_img_file_paths, k)
        for i_set in range(k):
            val_img_file_paths = list()
            train_img_file_paths = list()
            for j_split, split in enumerate(k_splits):
                if i_set == j_split:
                    val_img_file_paths.extend(split)
                else:
                    train_img_file_paths.extend(split)

            train_val_img_file_paths_k_sets[i_set]['train'].extend(train_img_file_paths)
            train_val_img_file_paths_k_sets[i_set]['val'].extend(val_img_file_paths)

    # 경로명에서 root dir 제거
    for i_set in range(k):
        train_val_img_file_paths_k_sets[i_set]['train'] = \
            [os.path.join(os.path.basename(os.path.dirname(full_path)), os.path.basename(full_path))
             for full_path in train_val_img_file_paths_k_sets[i_set]['train']]
        train_val_img_file_paths_k_sets[i_set]['val'] = \
            [os.path.join(os.path.basename(os.path.dirname(full_path)), os.path.basename(full_path))
             for full_path in train_val_img_file_paths_k_sets[i_set]['val']]

    test_img_file_paths = [os.path.join(os.path.basename(os.path.dirname(full_path)), os.path.basename(full_path))
                           for full_path in test_img_file_paths]

    print()
    print(f"# of train images of 1st set: {len(train_val_img_file_paths_k_sets[0]['train'])}")
    print(f"# of val images of 1st set: {len(train_val_img_file_paths_k_sets[0]['val'])}")
    print(f"# of test images of 1st set: {len(test_img_file_paths)}")

    for i_set in range(k):
        train_val_file_name = 'train_val_' + str(i_set + 1) + '_' + str(k)
        if in_args.seed:
            train_val_file_name += f'_seed{in_args.seed}'
        train_val_file_name += '.json'
        with open(os.path.join(dst_dir, train_val_file_name), 'w') as out_train_val_file:
            json.dump(train_val_img_file_paths_k_sets[i_set], out_train_val_file, indent=4)

    test_file_name = 'test'
    if in_args.seed:
        test_file_name += f'_seed{in_args.seed}'
    test_file_name += '.json'
    with open(os.path.join(dst_dir, test_file_name), 'w') as out_test_file:
        json.dump(test_img_file_paths, out_test_file, indent=4)
