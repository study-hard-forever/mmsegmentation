# 本文件仅为Add a new dataset class. 构建新数据集的部分，对于创建自定义配置文件与训练测试流程请参考demo/MMSegmentation_Tutorial.ipynb
# # Finetune a semantic segmentation model on a new dataset

'''
To finetune on a customized dataset, the following steps are necessary. 
1. Add a new dataset class. 
2. Create a config file accordingly. 
3. Perform training and evaluation. 
'''

# 1. Add a new dataset class. 
'''
本次实验选用的是斯坦福背景数据集stanford_background.tar.gz,解压缩完毕后为iccv09Data文件夹:
.
|-- README
|-- horizons.txt
|-- images
`-- labels
斯坦福背景数据集包含715张图像,这些图像选自现有的公开数据集LabelMe、MSRC、PASCAL VOC和Geometric Context。这些数据集中的图像主要是室外场景,每个图像大约包含320x240像素。
在这个教程中,我们使用区域注释作为标签。总共有8个类别,即天空、树木、道路、草地、水体、建筑、山峰和前景目标。
There are 8 classes in total, i.e. sky, tree, road, grass, water, building, mountain, and foreground object. 
'''

# Let's take a look at the dataset
import mmcv
import mmengine
import matplotlib.pyplot as plt

'''description：读取单张图像测试：
img = mmcv.imread('iccv09Data/images/6000124.jpg')
plt.figure(figsize=(8, 6))
plt.imshow(mmcv.bgr2rgb(img))
# plt.show()  # 远程：避免直接展示
plt.savefig('figure.png')
'''


'''description：
# 斯坦福背景数据集的标签文件是txt格式的，而不是图像格式，因此需要做格式转换
# We need to convert the annotation into semantic map format as an image.
# define dataset root and directory for images and annotations
'''

data_root = 'iccv09Data'
img_dir = 'images'
ann_dir = 'labels'
img_anno_dir = 'img_labels'  # 此处重新定义了img_labels的存放位置
# define class and palette for better visualization  # palette：调色板
classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]


import os.path as osp
import numpy as np
from PIL import Image

'''description：实际格式转换部分的代码：
# convert dataset annotation to semantic segmentation map
for file in mmengine.scandir(osp.join(data_root, ann_dir), suffix='.regions.txt'):
  seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
  
  seg_img = Image.fromarray(seg_map).convert('P')
  seg_img.putpalette(np.array(palette, dtype=np.uint8))
  """
  这段代码的作用是将一个数组 seg_map 转换成图像，并将调色板应用于图像。

  Image.fromarray(seg_map) 将 seg_map 数组转换为图像对象。
  .convert('P') 将图像转换为调色板模式，其中每个像素值都是一个索引，而不是直接的颜色值。
  seg_img.putpalette(np.array(palette, dtype=np.uint8)) 将调色板 palette 应用于图像，其中 palette 是一个包含颜色信息的数组。
  通过这些步骤，代码将 seg_map 数组转换为一个带有指定调色板的图像对象 seg_img。
  """

  seg_img.save(osp.join(data_root, img_anno_dir, file.replace('.regions.txt', 
                                                         '.png')))
'''


'''description：查看转换后的标签图像
# Let's take a look at the segmentation map we got
import matplotlib.patches as mpatches
img = Image.open('iccv09Data/img_labels/6000124.png')
plt.figure(figsize=(10, 6))  # 手动调整即可，默认为plt.figure(figsize=(8, 6))，但是由于宽度不够，显示不了完整的标签
im = plt.imshow(np.array(img.convert('RGB')))

# create a patch (proxy artist) for every color 
patches = [mpatches.Patch(color=np.array(palette[i])/255., 
                          label=classes[i]) for i in range(8)]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
           fontsize='large')
"""
这段代码是用于在使用matplotlib库时，为图表添加一个代表不同类别的图例，其中每个类别都有自己的颜色和标签。

- `patches` 是一个列表，它通过列表推导式创建了一系列的 `mpatches.Patch` 对象。每个 `Patch` 对象代表图例中的一个条目。
- `mpatches.Patch(color=np.array(palette[i])/255., label=classes[i])` 创建一个色块，颜色由 `palette` 数组中的第 `i` 个颜色决定，并且这个颜色被除以255，因为matplotlib期望的是0到1之间的RGB值。`label=classes[i]` 设置了每个色块的标签，这些标签由 `classes` 列表提供。
- `for i in range(8)` 表示循环创建8个这样的 `Patch` 对象，假设有8种不同的类别。
- `plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')` 将这些色块作为图例的句柄添加到图表中。参数 `bbox_to_anchor` 和 `loc` 控制图例框的位置，`borderaxespad` 是图例框边缘与图表边缘之间的距离，`fontsize` 设置了图例文字的大小。

总之，这段代码是用来创建和定位一个图例的，使得图表中的每种颜色都能与相应的类别名称相对应。
"""

# plt.show()
plt.savefig('iccv09Data_img_labels_6000124.png')
'''


'''description：随机划分训练集与测试集
# split train/val set randomly
split_dir = 'splits'
mmengine.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmengine.scandir(
    osp.join(data_root, img_anno_dir), suffix='.png')]
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list)*4/5)
  f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
  # select last 1/5 as train set
  f.writelines(line + '\n' for line in filename_list[train_length:])
'''

