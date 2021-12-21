"""
WBF (weighted box fusion) 방식의 앙상블
여러 모델(또는 앙상블)의 inference를 test.py의 --save-output 옵션을 이용하여 저장한 뒤,
본 script 실행시 입력하여 앙상블을 할 수 있다.
앙상블한 결과를 test.py에 --load-output-pickle 옵션을 이용해 입력하면, evaluation 할 수 있다.
"""
import argparse
import pickle
from typing import List, Tuple

import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import torch
from tqdm import tqdm


def get_input_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', nargs='+', help='pickle files of predictions(inferences)')
    parser.add_argument('--dst', type=str, help='path to save ensemble result to')
    parser.add_argument('--weights', nargs='+', default=[], help='weights of predictions(inferences)')
    parser.add_argument('--img-width', type=int, default=704,
                        help='image width, this value is used for normalizing boxes\' coordinates')
    parser.add_argument('--img-height', type=int, default=448,
                        help='image height, this value is used for normalizing boxes\' coordinates')
    parser.add_argument('--iou-thr', type=float, default=0.7, help='iou-thr of WBF')
    parser.add_argument('--skip-box-thr', type=float, default=0.0001, help='skip-box-thr of WBF')

    return parser.parse_args()


def load_outputs_and_group_by_image(pickle_file_path_list: List[str]) -> List[Tuple[torch.Tensor]]:
    """
    각 모델 또는 앙상블의 inference 결과가 저장된 pickle 파일을 로드한다.
    이미지 별로  각 inference의 box들 grouping 해놓는다.
    Args:
        pickle_file_path_list: inference 결과가 저장된 pickle file 경로 리스트,
        각 pickle 파일은 test.py에서 --save-output 옵션을 통해 저장할 수 있다.

    Returns: 이미지 별 각 inference의 box 들

    """
    model_results: List[List[torch.Tensor]] = []  # model - image - box
    for pickle_file_path in pickle_file_path_list:
        with open(pickle_file_path, 'rb') as pkl_file:
            model_results.append(pickle.load(pkl_file))

    model_results_by_image: List[Tuple[torch.Tensor]] = list(zip(*model_results))  # image - model - box
    return model_results_by_image


def ensemble(model_results_by_image: List[Tuple[torch.Tensor]],
             weights: List[float],
             img_width: int,
             img_height: int,
             iou_thr: float,
             skip_box_thr: float
             ) -> List[torch.Tensor]:
    """
    WBF (weighted box fusion) 방식으로 앙상블 한다.
    앙상블 결과를 test.py에서 저장 또는 로드하는 inference output과 같은 format으로 맞춰준다.
    Args:
        model_results_by_image:load_outputs_and_group_by_image 함수로 부터 얻을 수 있는 이미지 별 각 inference의 box 들
        weights: 각 inference 에 적용될 가중치
        img_width:
        img_height:
        iou_thr:
        skip_box_thr:

    Returns: 앙상블 결과 (test.py에서 저장 또는 로드하는 inference output과 같은 format)

    """
    normalize_divider = np.array([img_width, img_height, img_width, img_height], dtype=np.float32)
    unnormalize_factor = normalize_divider

    def get_boxes_in_a_image(yolo_preds: torch.Tensor) -> Tuple[List[np.ndarray], List[float], List[int]]:
        normalized_boxes: List[np.ndarray] = []
        scores: List[float] = []
        labels: List[int] = []
        yolo_preds_np = yolo_preds.cpu().numpy()
        for yolo_pred in yolo_preds_np:
            normalized_boxes.append(yolo_pred[:4] / normalize_divider)
            scores.append(yolo_pred[4])
            labels.append(int(yolo_pred[5]))

        return normalized_boxes, scores, labels

    def get_yolo_format_preds(normalized_boxes: List[np.ndarray], scores: List[float], labels: List[int]) \
            -> torch.Tensor:
        yolo_preds_list: List[np.ndarray] = []
        for box, score, label in zip(normalized_boxes, scores, labels):
            unnormalized_box = box * unnormalize_factor
            yolo_pred = np.array([*unnormalized_box, score, label], dtype=np.float32)
            yolo_preds_list.append(yolo_pred)

        return torch.tensor(yolo_preds_list)

    ensemble_results: List[torch.Tensor] = []
    for model_results_for_a_image in tqdm(model_results_by_image):
        boxes_list: List[List[np.ndarray]] = []
        scores_list: List[List[float]] = []
        labels_list: List[List[int]] = []
        for a_model_result_for_a_image in model_results_for_a_image:
            boxes, scores, labels = get_boxes_in_a_image(a_model_result_for_a_image)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        ensemble_boxes, ensemble_scores, ensemble_labels = \
            weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                  weights=weights,
                                  iou_thr=iou_thr,
                                  skip_box_thr=skip_box_thr)

        yolo_format_preds_for_a_image = get_yolo_format_preds(ensemble_boxes, ensemble_scores, ensemble_labels)
        ensemble_results.append(yolo_format_preds_for_a_image)

    return ensemble_results


def save(results: List[torch.Tensor], dst_path: str) -> None:
    """
    결과를 pickle 파일로 저장한다.
    pickle파일을 test.py 에 입력하여 evaluation 할 수 있다.
    Args:
        results: 앙상블 결과
        dst_path: 결과를 저장할 경로

    Returns: None
    """
    with open(dst_path, 'wb') as dst_pkl_file:
        pickle.dump(results, dst_pkl_file)
    print(f'Saved - {dst_path}')


if __name__ == '__main__':
    in_args = get_input_args()

    if not in_args.weights:
        in_args.weights = [1. for _ in range(len(in_args.preds))]
    else:
        in_args.weights = [float(weight) for weight in in_args.weights]

    print(in_args)

    preds_by_image = load_outputs_and_group_by_image(in_args.preds)
    ensemble_res = ensemble(preds_by_image,
                            in_args.weights,
                            in_args.img_width,
                            in_args.img_height,
                            in_args.iou_thr,
                            in_args.skip_box_thr)
    save(ensemble_res, in_args.dst)
