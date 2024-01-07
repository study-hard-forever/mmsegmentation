from PIL import Image
import glob
import os
from tqdm import tqdm
import numpy as np
import cv2

imgs_path = r"data/VOCdevkit/VOC2007/SegmentationClass"
# imgs_path = r'data/VOCdevkit/VOC2007/JPEGImages'

# 使用 glob.glob 匹配文件
# 返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。
file_pattern = "NG_?????.png"
matching_files = glob.glob(imgs_path + "/" + file_pattern)

count = 0  # 用于统计不符合规范的数据


def check_image_dimensions(image_path):
    if image_path in matching_files:  # 匹配到的图像才进行判断
        # print(f"Processing image: {image_path}")
        global count  # 使用 global 关键字声明全局变量
        try:
            with Image.open(image_path) as img:
                # img.verify()
                if img.mode == "L":
                    # print(img.mode)
                    pass
                else:
                    # print(img.mode)
                    # print(image_path)
                    count += 1

                    # 此处图像已经不是L灰度图像，而是RGB图像，获取图像的 RGB 数据：
                    # pixels = img.convert("RGB").getdata()

                    # # 判断每个像素的三个通道是否一致
                    # for r, g, b in pixels:
                    #     if r != g or r != b or g != b:
                    #         print('False')

                    grayscale_img = img.convert("L")
                    # 获取图像的数据以保存灰度图像
                    data = grayscale_img.getdata()
                    # print(list(data))
                    data = np.array(grayscale_img)
                    # 输出数据的类型和形状
                    # print("Data type:", data.dtype)
                    # print("Data shape:", data.shape)
                    # Image.fromarray(data, mode='L').save(new_path)  # 最初想要采用Image进行保存，但是发现卡住了（opencv也卡，这是由于上面的img.convert("RGB").getdata()代码造成的，由于进行了多次转换，中间存在部分bug，因此在转换时尽量采用copy的数据进行转换而不是从原对象上）
                    # 将数据类型转换为 uint8，并使用 cv2.imwrite 保存图像
                    cv2.imwrite(
                        image_path.replace("SegmentationClass", "Segmentation"),
                        data.astype(np.uint8),
                    )
                return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    else:  # 如果不在匹配的文件列表中也默认返回True
        return True


for i in tqdm(os.listdir(imgs_path)):
    image_path = os.path.join(imgs_path, i)

    if check_image_dimensions(image_path):
        # print("Image dimensions are valid.")
        pass
    else:
        print("Image dimensions are not valid.")
print(count)
