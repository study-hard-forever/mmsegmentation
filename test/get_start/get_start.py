'''
description：来自快速开始（安装并测试mmseg）
参考链接：
https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation
注：本次环境安装的是torch-cu118：
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

环境保存在了conda env export > environment.yml下
测试（注意文件路径地址）：

1、下载测试模型：
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
2、命令行直接预测：
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
结果将会保存为result.jpg
'''


'''description：3、代码预测方式（效果同命令行预测）
# 快速开始的代码（注意视频部分：mmcv.VideoReader('video.mp4')；mmcv对视频进行了抽帧以便对每一帧进行语义分割）
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/demo.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_model(model, img)
# visualize the results in a new window
show_result_pyplot(model, img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
# 你可以使用 (0, 1] 的范围来改变绘制的分割地图的不透明度。
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
   result = inference_model(model, frame)
   show_result_pyplot(model, frame, result, wait_time=1)
'''


'''description：模型推理
from mmseg.apis import MMSegInferencer
# Load models into memory
inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')
# Inference
# inferencer('demo/demo.png', show=True)  # 直接显示可视化结果

# Save visualized rendering color maps and predicted results
# out_dir is the directory to save the output results, img_out_dir and pred_out_dir are subdirectories of out_dir
# to save visualized rendering color maps and predicted results
# out_dir是保存输出结果的目录，img_out_dir和pred_out_dir是out_dir的子目录，用于保存可视化的渲染颜色映射和预测结果
inferencer('demo/demo.png', out_dir='outputs', img_out_dir='vis', pred_out_dir='pred')  # 将结果保存到文件夹下
'''


'''description：展示所有模型列表
from mmseg.apis import MMSegInferencer
# models is a list of model names, and them will print automatically
models = MMSegInferencer.list_models('mmseg')
'''


'''description：模型推理的封装形式不同：
mmseg/apis/__init__.py

from .inference import inference_model, init_model, show_result_pyplot
from .mmseg_inferencer import MMSegInferencer

from .inference import inference_model, init_model, show_result_pyplot从函数角度一一构建模型推理的过程；代表代码如下：
demo/inference_demo.ipynb
demo/image_demo.py

from .mmseg_inferencer import MMSegInferencer从类角度直接构建推理过程；代表代码如下：
demo/image_demo_with_inferencer.py
'''


'''description：revert_sync_batchnorm将模型中所有的SyncBatchNorm（SyncBN）和mmcv.ops.sync_bn.SyncBatchNorm（MMSyncBN）层转换为BatchNormXd层：
demo/inference_demo.ipynb
在模型推理时用到了模块转换函数：

from mmengine.model.utils import revert_sync_batchnorm

# test a single image
img = 'demo.png'
if not torch.cuda.is_available():
    model = revert_sync_batchnorm(model)
result = inference_model(model, img)

revert_sync_batchnorm这是一个帮助转换的函数，用于将模型中所有的SyncBatchNorm（SyncBN）和mmcv.ops.sync_bn.SyncBatchNorm（MMSyncBN）层转换为BatchNormXd层。

为什么需要转换：
Since we use only one GPU, BN is used instead of SyncBN
demo/MMSegmentation_Tutorial.ipynb给出了解释：仅在单张GPU上时不需要使用SyncBN

转换`SyncBatchNorm`（同步批量归一化）层为`BatchNormXd`（批量归一化）层的原因通常与模型训练和部署环境的兼容性有关。以下是一些可能的原因：

1. **设备兼容性**：在某些设备上可能不支持同步批量归一化（SyncBN），或者该操作的效率不如批量归一化（BatchNorm）。

2. **多GPU训练与单GPU推理**：在多GPU训练时，SyncBN 可以在所有GPU之间同步均值和方差，以保证每个批次数据的一致性。但在单GPU推理时，这种同步是不必要的，使用标准的 BatchNorm 会更加高效。

3. **部署简化**：BatchNormXd 层通常在各种深度学习框架和硬件上得到了更广泛的支持和优化，转换为 BatchNormXd 可以简化模型的部署过程。

4. **性能优化**：BatchNormXd 在某些情况下可能比 SyncBN 有更好的性能，尤其是在推理时。

5. **框架要求**：有些深度学习框架或库可能不支持 SyncBN，或者在特定版本中存在兼容性问题，因此需要转换为 BatchNormXd 来确保模型能够在该框架上运行。

在实际应用中，开发者需要根据具体的应用场景和需求来决定是否进行这种转换。

上述是在CPU环境下进行推理，因此需要将`SyncBatchNorm`（同步批量归一化）层转换为`BatchNormXd`（批量归一化）层
'''

'''description：demo/video_demo.py是视频相关的内容，待查看

'''