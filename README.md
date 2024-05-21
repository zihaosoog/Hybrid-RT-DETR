# Hybrid-RT-DETR
 基于多尺度增强辅助编解码网络的无人机目标检测算法

Keywords：DETR；多尺度特征融合；辅助目标查询

## Results on VisDrone2019

| Method             | mAP   | $AP_{50}$ | $AP_{75}$ | $AP_s$ | $AP_m$ | $AP_l$ |
|--------------------|--------|------------|------------|--------|--------|--------|
| YOLOv5m6           | 27.00  | 44.4       | —          | —      | —      | —      |
| YOLOv6m            | 20.90  | 35.50      | 20.81      | 10.43  | 33.21  | 57.64  |
| YOLOv7             | 28.74  | 49.91      | —          | —      | —      | —      |
| YOLOv8m            | 27.33  | 44.71      | —          | —      | —      | —      |
| DAB-deformable-detr-R50 | 25.72  | 43.51      | 25.73      | 17.30  | 36.74  | 45.82  |
| DN-DAB-deformable-detr-R50 | 27.93  | 46.00      | 28.41      | 19.14  | 39.67  | 54.31  |
| Lite-DINO-H3L1-R50    | 32.79  | **55.21**      | 32.70      | **23.38**  | 44.65  | 58.81  |
| RT-DETR-R18         | 28.93  | 48.29      | 29.21      | 19.55  | 40.43  | 61.00  |
| RT-DETR-R50         | 32.20  | 52.85      | 32.65      | 22.27  | 44.87  | **68.43**  |
| 本章算法-R18           | 30.57  | 50.78      | 30.91      | 21.40  | 42.38  | 59.70  |
| 本章算法-R50           | **33.14**  | **54.43**      | **33.89**      | **23.62**  | **45.94**  | 63.08  |


## Quick start
<details>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```
</details>

<details>
<summary>Data</summary>
 
Download VisDrone and convert it to COCO format annonations of train and val data.
</details>

<details>
<summary>Training & Evaluation</summary>

- Training on a Single GPU:

```shell
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- Training on Multiple GPUs:

```shell
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- Evaluation on Multiple GPUs:

```shell
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```
</details>

Tips: set `remap_mscoco_category: False`.
