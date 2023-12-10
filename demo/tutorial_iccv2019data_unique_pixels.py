# -*- coding: utf-8 -*-
# @Author      : MJG
# FileName     : 
# @Time        : 
# @Description : 多线程统计像素值
'''
经统计iccv09Data/img_labels生成的标签数据含像素值为调色板中规定的像素值(test/get_start/tutorial.py):
实际统计的标签像素值：
[0, 11, 12, 20, 25, 34, 38, 51, 53, 69, 81, 118, 120, 122, 123, 125, 127, 128, 129, 134, 241]
调色板中像素值：
[  0  11  12  20  25  34  38  51  53  69  81 118 120 122 123 125 127 128 129 134 241]
'''

import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

mask_path = r"iccv09Data/img_labels"

# 统计mask像素值
mask_category_set = set()
def unique_pixels(mask_file):
  # mask = cv2.imdecode(np.fromfile(mask_file, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
  mask = cv2.imdecode(np.fromfile(mask_file, dtype=np.uint8), cv2.IMREAD_COLOR)  # 此处是3通道标签
  # 查看mask中都有哪些像素
  unique_pixels = np.unique(mask)
  mask_category_set.update(unique_pixels)
  print(mask_file, unique_pixels)
     
if __name__ == "__main__": 
  with ThreadPoolExecutor() as executor:
    # 遍历文件夹中的所有文件，提交每个文件的处理任务到线程池
    futures = {executor.submit(unique_pixels, os.path.join(mask_path, mask_file)): mask_file for mask_file in os.listdir(mask_path)}

    # 等待所有任务完成
    for future in tqdm(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error processing file: {futures[future]} - {e}")

  # Convert the set to a sorted list
  sorted_list = sorted(mask_category_set)
  # Display or use the sorted list
  print(sorted_list)
    
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]
unique_pixels = np.unique(palette)
print(unique_pixels)

