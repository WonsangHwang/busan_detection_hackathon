# 2021 부산광역시 인공지능 학습용 데이터 해커톤 대회<br/>- 제 2 분야: 공원내 발생하는 불법객체 데이터를 활용한 모델 개발
대회 규정에 따라 사용가능한 모델은 Yolo v4로 제한됨<br>
[PyTorch implementation of YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4) 를 fork 후 추가, 수정, 튜닝하여 개발함


## Requirements
- System
  - Ubuntu 18.04 에서 개발됨
  - GPU VRAM 12.8 GB 이상<br/>(아래의 실험들은 image size 640, batch size 16으로 이루어짐. VRAM 부족시 이 수치들은 하향 조정되어야 함)
  - RAM 12.1 GB 이상 <br/>(만족할 수 없는 경우, train, test시 --cache-image를 사용하지 말 것) 

- Python
  - Python >= 3.7
  - ```shell
    pip install -r requirements.txt
    ```

  
## Train / Val / Test 용 데이터 분할 및 Yolo형식으로 입력 데이터 변환 
- Train / Val / Test 구분
  - 데이터 들을 Train / Val / Test 용도로 구분하고, 각 용도로 쓰여질 파일 리스트를 json 형식으로 저장한다.
  - ```shell
    python convert2yolo/split_data.py --src /opt/ml/busan_detection_data --dst data/busan --test-ratio 0.1 --k 9
    ```
    - 위 예시의 경우, data/busan에 9개의 train/val set을 구성하고, 각 set 별로 json이 저장된다. 또한 1개의 test용 파일리스트를 갖는 json이 저장된다.
    - 위 예시의 경우, train:val:test = 8:1:1 비율 데이터가 구성된다.
    - 옵션 설명
      - src: 대회 제공 원본 데이터가 저장된 경로
      - dst: 파일 리스트 json이 저정될 경로
      - test-ratio: 전체 데이터 중 test용으로 쓰일 비율
      - k: test용 이외의 데이터는 k-fold 방식으로 train/val로 나뉜다. 그 때의 k
      - seed: 데이터 shuffle 시 쓰이는 seed
    
- Yolo 형식 데이터 생성
  - Yolo 형식의 디렉토리 구조를 생성한다.
  - 생성된 디렉토리 구조에 이미지 파일을 복사한다.
  - Yolo 형식의 annotation 파일을 생성하여 저장한다.
  - ```shell
     python convert2yolo/make_yolo_data_dir.py --train data/busan/train_val_1_9.json --test data/busan/test.json --src /opt/ml/busan_detection_data --dst /opt/ml/busan_detection_data_yolo_1_9
    ```
    - 생성된 'dst' 디렉토리는 data yaml파일에 입력되어 train, test시에 이용되어진다.
    - 옵션 설명
      - train: split_data.py 통해 생성된 train/val set 중 선택된 set의 파일 리스트 json 
      - test: split_data.py 통해 생성된 test 파일 리스트 json
      - src: 대회 제공 원본 데이터가 저장된 경로
      - dst: Yolo 형식 데이터가 저장될 경로


## 실험
- 모든 실험은 공통적으로 아래 조건에서 수행되었다.
  - csp-x-leaky 모델 사용
  - image size: 640 (train, val, test)
  - train epochs: 100
  - batch size: 16
- 그 외, 아래 실험에서 명시하지 않은 사항은 yolov4 기본 세팅을 따랐다.

### Augmentation

|실험 번호|fliplr|trans-late|mosaic|hsv|rotate|scale|persp-ective|mixup|AP<sup>val</sup>| AP<sub>50</sub><sup>val</sup>|AP<sup>test</sup>| AP<sub>50</sub><sup>test</sup>|비고|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|A1|✔|✔|✔| | | | | |0.643|0.792|0.652|0.802|baseline|
|A2|✔|✔|✔|✔| | | | |0.645|0.794|0.648|0.801| |
|A3|✔|✔|✔| |✔| | | |0.630|0.811|0.635|0.816|서버 문제로 94 epoch까지만 학습|
|A4|✔|✔|✔| | |✔| | |0.668|0.821|0.675|0.829| |
|A5|✔|✔|✔| | | |✔| |0.643|0.821|0.660|0.830| |
|A6|✔|✔|✔| | | | |✔|0.664|0.816|0.669|0.822| |
|**A7**|✔|✔|✔|✔| |✔| | |**0.673**|**0.826**|**0.681**|**0.832**|yolo default|
|A8|✔|✔|✔|✔| |✔|✔|✔|0.603|0.810|0.624|0.818| |
|A9|✔|✔|✔|✔|✔|✔|✔|✔|0.585|0.810|0.602|0.820| |

- 실험 A1 - flip lr, translate, mosaic
  - Train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_fliplr_t_mosaic.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights '' --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_100e
    ```
  - Test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_flip1r_t_mosaic_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_100e_best_ap
    ```
- 실험 A2 - flip lr, translate, mosaic, hsv
  - Train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_fliplr_t_mosaic_hsv.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights '' --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_hsv_100e
    ```
  - Test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_flip1r_t_mosaic_hsv_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_hsv_100e_best_ap
    ```
- 실험 A3 - flip lr, translate, mosaic, rotate
  - Train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_fliplr_t_mosaic_rotate.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights '' --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_rotate_100e
    ```
  - test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_flip1r_t_mosaic_rotate_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_rotate_100e_best_ap
    ```
- 실험 A4 - flip lr, translate, mosaic, scale
  - Train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_fliplr_t_mosaic_scale.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights '' --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_scale_100e
    ```
  - Test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_flip1r_t_mosaic_scale_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_scale_100e_best_ap
    ```
- 실험 A5 - flip lr, translate, mosaic, perspective
  - Train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_fliplr_t_mosaic_per.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights '' --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_per_100e
    ```
  - Test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_flip1r_t_mosaic_per_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_per_100e_best_ap
    ```
- 실험 A6 - flip lr, translate, mosaic, mixup
  - Train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_fliplr_t_mosaic_mixup.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights '' --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_mixup_100e
    ```
  - Test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_flip1r_t_mosaic_mixup_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_mixup_100e_best_ap
    ```
- 실험 A7 - flip lr, translate, mosaic, hsv, scale
  - Train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_yolo_default.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights '' --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_hsv_scale_yolo_default_100e
    ```
  - Test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_flip1r_t_mosaic_hsv_scale_yolo_default_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_flip1r_t_mosaic_hsv_scale_yolo_default_100e_best_ap
    ```
- 실험 A8 - flip lr, translate, mosaic, hsv, scale, perspective, mixup
  - Train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_all_aug_except_rotate.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights '' --project busan --name x-leaky_bs16_640_640_all_aug_except_rotate_100e
    ```
  - Test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_all_aug_except_rotate_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_all_aug_except_rotate_100e_best_ap
    ```
- 실험 A9 - flip lr, translate, mosaic, hsv, rotate, scale, perspective, mixup
  - Train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_all_aug.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights '' --project busan --name x-leaky_bs16_640_640_all_aug_100e
    ```
  - Test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_all_aug_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_all_aug_100e_best_ap
    ```

### Focal Loss
Object loss, classification loss에 focal loss를 적용

|실험 번호|실험 내용|AP<sup>val</sup>| AP<sub>50</sub><sup>val</sup>|AP<sup>test</sup>| AP<sub>50</sub><sup>test</sup>|비고|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**F1**|**CE loss**|**0.693**|**0.856**|**0.691**|**0.860**|yolo default|
|F2|Focal loss|0.639|0.797|0.644|0.803|gamma=1.5|

- 실험 F1 - CE loss
  - train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_yolo_default.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --weights '' --project busan --name x-leaky_bs16_640_640_ce_loss_100e_13c
    ```
  - test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_ce_loss_100e_13c/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_ce_loss_100e_13c_best_ap
    ```
  - 실험 F2 - focal loss
    - train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_1_9.yaml --hyp data/busan/hyp_yolo_default_focal.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --weights '' --project busan --name x-leaky_bs16_640_640_focal_loss_100e_13c
    ```
  - test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_1_9.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_focal_loss_100e_13c/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_focal_loss_100e_13c_best_ap
    ```

### K-Fold
k=9 이므로 9개의 train/val set이 있으나, 시간 관계상 5개에 대해서만 train 하여 ensemble 하고자 한다.<br/>
아래 실험에서, data set 변경 외에는 다음 조건이 공통적으로 적용되었다. 
- Augmentation ~ flip lr, translate, mosaic, hsv, scale (yolo default) 적용
- CE Loss 적용

|실험 번호|실험 내용|AP<sup>val</sup>| AP<sub>50</sub><sup>val</sup>|AP<sup>test</sup>| AP<sub>50</sub><sup>test</sup>|비고|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|K1|1번 train/val set|0.693|0.856|0.691|0.860|F1 실험과 동일|
|K2|2번 train/val set|0.688|0.856|0.686|0.858| | 
|K3|3번 train/val set|0.697|0.862|0.687|0.859| | 
|K4|4번 train/val set|0.690|0.858|0.694|0.864| |
|K5|5번 train/val set|0.690|0.855|0.690|0.865| |

- 실험 F1 - 1번 train/val set  ➜ F1 실험과 동일

- 실험 F2 - 2번 train/val set
  - train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_2_9.yaml --hyp data/busan/hyp_yolo_default.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --weights '' --project busan --name x-leaky_bs16_640_640_k2_100e
    ```
  - test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_2_9.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_k2_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_k2_100e
    ```

- 실험 F3 - 3번 train/val set
  - train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_3_9.yaml --hyp data/busan/hyp_yolo_default.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --weights '' --project busan --name x-leaky_bs16_640_640_k3_100e
    ```
  - test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_3_9.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_k3_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_k3_100e
    ```

- 실험 F4 - 4번 train/val set
  - train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_4_9.yaml --hyp data/busan/hyp_yolo_default.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --weights '' --project busan --name x-leaky_bs16_640_640_k4_100e
    ```
  - test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_4_9.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_k4_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_k4_100e
    ```

- 실험 F5 - 5번 train/val set
  - train
    ```shell
    python train.py --device 0 --cache-images --batch-size 16 --epochs 100 --img-size 640 640 --data data/busan/park_5_9.yaml --hyp data/busan/hyp_yolo_default.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --weights '' --project busan --name x-leaky_bs16_640_640_k5_100e
    ```
  - test
    ```shell
    python test.py --device 0 --task test --batch 16 --img 640  --data data/busan/park_5_9.yaml --cfg cfg/yolov4-csp-x-leaky_busan.cfg --names data/busan/park.names --weights busan/x-leaky_bs16_640_640_k5_100e/weights/best_ap.pt --project busan --name x-leaky_bs16_640_640_k5_100e
    ```
  
### Ensemble - WBF (Weighted Box Fusion)
|실험 번호|대상 inference|IOU threshold|Weights|AP<sup>test</sup>| AP<sub>50</sub><sup>test</sup>|비고|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|E1|K1~K5|0.60|1,1,1,1,1|0.688|0.860| |
|E2|K1~K5|0.65|1,1,1,1,1|0.709|0.870| |
|**E3**|K1~K5|**0.70**|**1,1,1,1,1**|**0.714**|**0.873**| |
|E4|K1~K5|0.75|1,1,1,1,1|0.712|0.872| |
|E5|K1~K5|0.80|1,1,1,1,1|0.710|0.869| |
|E6|K1~K5|0.70|1,1,1,2.0,1|0.712|0.872| |
|E7|K1~K5|0.70|1,1,1,3.0,1|0.709|0.871| |
|**E8**|K1~K5|**0.70**|**1.5,1,1,2.0,1.5**|**0.714**|**0.873**| |
|**E9**|K1~K5|**0.70**|**1.5,1.2,1.2,2.0,1.5**|**0.714**|**0.873**| |

- 실험 E1
  - ensemble
  ```shell
  python ensemble.py --dst busan/ensemble_iou60_w11111.pkl --iou-thr 0.6 --preds busan/test_best_ap_x-leaky_bs16_640_640_k1_100e/test_best_ap_x-leaky_bs16_640_640_k1_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k2_100e/test_best_ap_x-leaky_bs16_640_640_k2_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k5_100e/test_best_ap_x-leaky_bs16_640_640_k5_100e_output.pkl
  ```
  - test
  ```shell
  python test.py --device 0 --batch 16 --img 640  --data data/busan/park_1_9.yaml --names data/busan/park.names --load-output-pickle busan/ensemble_iou60_w11111.pkl --plots --task test --project busan --name ensemble_iou60_w11111
  ```
  
- 실험 E2
  - ensemble
  ```shell
  python ensemble.py --dst busan/ensemble_iou65_w11111.pkl --iou-thr 0.65 --preds busan/test_best_ap_x-leaky_bs16_640_640_k1_100e/test_best_ap_x-leaky_bs16_640_640_k1_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k2_100e/test_best_ap_x-leaky_bs16_640_640_k2_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k5_100e/test_best_ap_x-leaky_bs16_640_640_k5_100e_output.pkl
  ```
  - test
  ```shell
  python test.py --device 0 --batch 16 --img 640  --data data/busan/park_1_9.yaml --names data/busan/park.names --load-output-pickle busan/ensemble_iou65_w11111.pkl --plots --task test --project busan --name ensemble_iou65_w11111
  ```

- 실험 E3
  - ensemble
  ```shell
  python ensemble.py --dst busan/ensemble_iou70_w11111.pkl --iou-thr 0.7 --preds busan/test_best_ap_x-leaky_bs16_640_640_k1_100e/test_best_ap_x-leaky_bs16_640_640_k1_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k2_100e/test_best_ap_x-leaky_bs16_640_640_k2_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k5_100e/test_best_ap_x-leaky_bs16_640_640_k5_100e_output.pkl
  ```
  - test
  ```shell
  python test.py --device 0 --batch 16 --img 640  --data data/busan/park_1_9.yaml --names data/busan/park.names --load-output-pickle busan/ensemble_iou70_w11111.pkl --plots --task test --project busan --name ensemble_iou70_w11111
  ```
  
- 실험 E4
  - ensemble
  ```shell
  python ensemble.py --dst busan/ensemble_iou75_w11111.pkl --iou-thr 0.75 --preds busan/test_best_ap_x-leaky_bs16_640_640_k1_100e/test_best_ap_x-leaky_bs16_640_640_k1_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k2_100e/test_best_ap_x-leaky_bs16_640_640_k2_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k5_100e/test_best_ap_x-leaky_bs16_640_640_k5_100e_output.pkl
  ```
  - test
  ```shell
  python test.py --device 0 --batch 16 --img 640  --data data/busan/park_1_9.yaml --names data/busan/park.names --load-output-pickle busan/ensemble_iou75_w11111.pkl --plots --task test --project busan --name ensemble_iou75_w11111
  ```

- 실험 E5
  - ensemble
  ```shell
  python ensemble.py --dst busan/ensemble_iou80_w11111.pkl --iou-thr 0.8 --preds busan/test_best_ap_x-leaky_bs16_640_640_k1_100e/test_best_ap_x-leaky_bs16_640_640_k1_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k2_100e/test_best_ap_x-leaky_bs16_640_640_k2_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k5_100e/test_best_ap_x-leaky_bs16_640_640_k5_100e_output.pkl
  ```
  - test
  ```shell
  python test.py --device 0 --batch 16 --img 640  --data data/busan/park_1_9.yaml --names data/busan/park.names --load-output-pickle busan/ensemble_iou80_w11111.pkl --plots --task test --project busan --name ensemble_iou80_w11111
  ```

- 실험 E6
  - ensemble
  ```shell
  python ensemble.py --dst busan/ensemble_iou70_w1010102010.pkl --iou-thr 0.7 --weights 1.0 1.0 1.0 2.0 1.0 --preds busan/test_best_ap_x-leaky_bs16_640_640_k1_100e/test_best_ap_x-leaky_bs16_640_640_k1_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k2_100e/test_best_ap_x-leaky_bs16_640_640_k2_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k5_100e/test_best_ap_x-leaky_bs16_640_640_k5_100e_output.pkl
  ```
  - test
  ```shell
  python test.py --device 0 --batch 16 --img 640  --data data/busan/park_1_9.yaml --names data/busan/park.names --load-output-pickle busan/ensemble_iou70_w1010102010.pkl --plots --task test --project busan --name ensemble_iou70_w1010102010
  ```
  
- 실험 E7
  - ensemble
  ```shell
  python ensemble.py --dst busan/ensemble_iou70_w1010103010.pkl --iou-thr 0.7 --weights 1.0 1.0 1.0 3.0 1.0 --preds busan/test_best_ap_x-leaky_bs16_640_640_k1_100e/test_best_ap_x-leaky_bs16_640_640_k1_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k2_100e/test_best_ap_x-leaky_bs16_640_640_k2_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k5_100e/test_best_ap_x-leaky_bs16_640_640_k5_100e_output.pkl
  ```
  - test
  ```shell
  python test.py --device 0 --batch 16 --img 640  --data data/busan/park_1_9.yaml --names data/busan/park.names --load-output-pickle busan/ensemble_iou70_w1010103010.pkl --plots --task test --project busan --name ensemble_iou70_w1010103010
  ```

- 실험 E8
  - ensemble
  ```shell
  python ensemble.py --dst busan/ensemble_iou70_w1510102015.pkl --iou-thr 0.7 --weights 1.5 1.0 1.0 2.0 1.5 --preds busan/test_best_ap_x-leaky_bs16_640_640_k1_100e/test_best_ap_x-leaky_bs16_640_640_k1_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k2_100e/test_best_ap_x-leaky_bs16_640_640_k2_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k5_100e/test_best_ap_x-leaky_bs16_640_640_k5_100e_output.pkl
  ```
  - test
  ```shell
  python test.py --device 0 --batch 16 --img 640  --data data/busan/park_1_9.yaml --names data/busan/park.names --load-output-pickle busan/ensemble_iou70_w1510102015.pkl --plots --task test --project busan --name ensemble_iou70_w1510102015
  ```

- 실험 E9
  - ensemble
  ```shell
  python ensemble.py --dst busan/ensemble_iou70_w1512122015.pkl --iou-thr 0.7 --weights 1.5 1.2 1.2 2.0 1.5 --preds busan/test_best_ap_x-leaky_bs16_640_640_k1_100e/test_best_ap_x-leaky_bs16_640_640_k1_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k2_100e/test_best_ap_x-leaky_bs16_640_640_k2_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k3_100e/test_best_ap_x-leaky_bs16_640_640_k3_100e_output.pkl busan/test_best_ap_x-leaky_bs16_640_640_k5_100e/test_best_ap_x-leaky_bs16_640_640_k5_100e_output.pkl
  ```
  - test
  ```shell
  python test.py --device 0 --batch 16 --img 640  --data data/busan/park_1_9.yaml --names data/busan/park.names --load-output-pickle busan/ensemble_iou70_w1512122015.pkl --plots --task test --project busan --name ensemble_iou70_w1512122015
  ```


