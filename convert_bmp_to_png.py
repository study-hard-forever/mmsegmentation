from PIL import Image
import os
from tqdm import tqdm

def convert_bmp_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
            
    # 遍历输入文件夹中的所有 BMP 文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".bmp"):
            # 构造输入和输出文件的完整路径
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)

            # 打开 BMP 图像并保存为 PNG 格式
            with Image.open(input_path) as img:
                img.save(output_path, "PNG")

if __name__ == "__main__":
    # 替换为你的输入和输出文件夹路径
    input_folder = "data/VOCdevkit/VOC2007/JPEGImages"
    output_folder = "data/VOCdevkit/VOC2007/JPEGImages"

    convert_bmp_to_png(input_folder, output_folder)
