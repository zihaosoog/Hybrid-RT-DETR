import json
# 读取文件内容并解析为字典列表

import os
import shutil

def best_stat(log_dir):

    # log_dir = r'C:\Users\zihao\Desktop\ablation\o2o\rtdetr_r18vd_6x_visdrone_o2o_twice_1\log.txt'

    with open(log_dir, 'r') as file:
        lines = file.readlines()

    # 将每一行转换为字典
    data = [json.loads(line) for line in lines]

    # 提取每个epoch的test_coco_eval_bbox列表
    test_coco_eval_bbox_lists = [d['test_coco_eval_bbox'] for d in data]

    maplist = [d[0] for d in test_coco_eval_bbox_lists]

    # best_res = maplist.index(max(maplist[:71] if len(maplist) >= 72 else maplist))

    best_res = maplist.index(max(maplist))

    # 提取对应的epoch值
    epochs = [d['epoch'] for d in data]

    # 打印结果
    # for epoch, coco_eval_list in zip(epochs, test_coco_eval_bbox_lists):
    print(f'log {os.path.split(log_dir)[-1]} Best Epoch {epochs[best_res]}: {test_coco_eval_bbox_lists[best_res]}', end='\n')
    

def pro(source_folders_path, destination_path):


    # 源文件夹路径，这里假设您的文件夹都在'./source_folders'路径下
    # source_folders_path = './source_folders'

    # 目标路径，您想要将重命名后的文件复制到的路径
    # destination_path = './destination_folder'

    # 确保目标路径存在
    os.makedirs(destination_path, exist_ok=True)

    # 遍历源文件夹路径下的所有文件夹
    for folder_name in os.listdir(source_folders_path):
        folder_path = os.path.join(source_folders_path, folder_name)
        # 检查是否是文件夹
        if os.path.isdir(folder_path):
            # 构建源log.txt文件的路径
            log_file_source = os.path.join(folder_path, 'log.txt')
            # 检查log.txt文件是否存在
            if os.path.exists(log_file_source):
                # 构建目标文件的路径，使用文件夹名称作为文件名
                log_file_destination = os.path.join(destination_path, folder_name + '.txt')
                # 复制并重命名文件
                shutil.copy2(log_file_source, log_file_destination)

    print("文件复制和重命名完成。")

def all_txtlog(target_folder_path):
    txt_files = [f'{target_folder_path}\\{f}' for f in os.listdir(target_folder_path) if f.endswith('.txt')]
    return txt_files

if __name__ == "__main__":
    # source_folders_path = destination_path = r'F:\rtdetr\experiments\hyp\spatial_select\source'
    # destination_path = r'F:\rtdetr\experiments\hyp\spatial_select'
    # pro(source_folders_path, destination_path)
    target_folder_path = r'F:\rtdetr\experiments\vssota\uavdt'
    all_logs = all_txtlog(target_folder_path)
    res = [best_stat(log_dir) for log_dir in all_logs]
    # print(res)