import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import colorsys
from src.core import YAMLConfig
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path
import time

import numpy as np

_SMALL_OBJECT_AREA_THRESH = 1000



class ImageReader:
    def __init__(self, resize=(640,640), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            # transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
            #     (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.resize = resize
        self.pil_img = None   #保存最近一次读取的图片的pil对象

    def __call__(self, image_path, *args, **kwargs):
        """
        读取图片
        """
        self.ori_img = Image.open(image_path).convert('RGB')
        self.pil_img = self.ori_img.resize((self.resize[0], self.resize[1]))
        
        return self.transform(self.pil_img).unsqueeze(0)


class Model(nn.Module):
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu') 
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        # print(self.postprocessor.deploy_mode)
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)

def generate_distinct_colors(k):
    colors = []
    # HSV色相的范围是0到1，色相间隔为1/k
    hue_interval = 1.0 / k
    for i in range(k):
        # 计算当前颜色的色相
        hue = i * hue_interval
        # 将HSV颜色转换为RGB颜色，饱和度和亮度设为最大
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # 将RGB颜色转换为整数格式
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        colors.append((r, g, b))
    return colors

def overlay_bbox_cv(img, dets, class_names, score_thresh, draw):
    all_box = []
    scores, labels, bboxs = dets
    for (score, label, bbox) in zip(scores, labels, bboxs):
        if score > score_thresh:
            x0, y0, x1, y1 = [i/640 for i in bbox]
            x0, x1 = x0 * img.size[0], x1 * img.size[0]
            y0, y1 = y0 * img.size[1], y1 * img.size[1]
            all_box.append([int(label.item())+1, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        color = tuple(_COLORS[label].astype(int).tolist())
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)

        # 在Pillow中绘制矩形
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        # 计算文本大小和位置
        font = ImageFont.load_default()
        txt_size = draw.textsize(text, font)
        draw.rectangle([x0, y0 - txt_size[1] - 1, x0 + txt_size[0] + txt_size[1], y0 - 1], fill=color)
        draw.text([x0, y0 - txt_size[1]], text, fill=txt_color, font=font)
        
    return img

from pathlib import Path

def get_image_files_in_folder(folder_path):
    """返回文件夹中所有图片文件的列表。"""
    current_directory = os.getcwd()
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif')
    image_files = [f for f in Path(f'{current_directory}/{folder_path}').iterdir() if f.suffix.lower() in image_extensions]
    return [image_file for image_file in image_files]


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtdetr/rtdetr_r18vd_6x_visdrone.yml", help="配置文件路径")
    parser.add_argument("--ckpt", default="r18_best.pth", help="权重文件路径")
    parser.add_argument("--image", default="inferes/images", help="待推理图片文件夹路径")
    parser.add_argument("--output_dir", default="inferes/dets_base", help="输出文件保存路径")
    parser.add_argument("--device", default="cpu")

    return parser


def main(args):
    mscoco_category2name = {
        1: 'pedestrian',
        2: 'people',
        3: 'bicycle',
        4: 'car',
        5: 'van',
        6: 'truck',
        7: 'tricycle',
        8: 'awning-tricycle',
        9: 'bus',
        10: 'motor',}

    img_path = Path(args.image)
    device = torch.device(args.device)
    reader = ImageReader(resize=(640,640))
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)

    img = reader(img_path).to(device)
    size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)
    start = time.time()
    output = model(img, size)
    print(f"推理耗时：{time.time() - start:.4f}s")
    labels, boxes, scores = output
    im = reader.ori_img
    draw = ImageDraw.Draw(im)

    thrh = 0.6
    
    im = overlay_bbox_cv(im, (scores[0], labels[0], boxes[0]), mscoco_category2name, thrh, draw)

    save_path = Path(args.output_dir) / img_path.name
    # save_path = f'{args.output_dir}/{img_path.name}'
    # cv2.imwrite(save_path,im)
    im.save(save_path)
    print(f"检测结果已保存至:{save_path}")

_COLORS = np.array([
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
    [0.635, 0.078, 0.184], [0.300, 0.300, 0.300], [0.600, 0.600, 0.600],
    [1.000, 0.000, 0.000], [1.000, 0.500, 0.000], [0.749, 0.749, 0.000],
    [0.000, 1.000, 0.000], [0.000, 0.000, 1.000], [0.667, 0.000, 1.000],
    [0.333, 0.333, 0.000]
]) * 255


if __name__ == "__main__":

    parser = get_argparser()
    args = parser.parse_args()
    
    # Instead of processing a single image, process all images in the folder
    images = get_image_files_in_folder(args.image)  # args.image now expects a folder path

    for img_path in images:
        args.image = img_path  # Update the image path for each iteration
        main(args)

    

