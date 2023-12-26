# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_model, init_model, show_result_pyplot

import mmcv

import cv2
import os.path as osp

'''
命令：
CUDA_VISIBLE_DEVICES=0 python image_inference.py \
  configs/ac/mask2former_beitv2_adapter_large_896_80k_ac_ms.py  \
  work_dirs/mask2former_beitv2_adapter_large_896_80k_ac_ms/best_mIoU_iter_72000.pth  \
  data/VOCdevkit/VOC2007/test_jpg/1_bengbianA_2_bengbianA_srcTray_1_srcIndex_1_DL_result_0_0_3_BengBian.jpg \
  --palette ac  \
  --out /home/sylu/workspace/mjg/nelson/data/ViT-Adapter_test_230713_Boston_AC_Test_NG
'''
def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', help='Image file')
    parser.add_argument('--out', type=str, default="results_iter_80000", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    '''
    args.out = "demo"  # 单张测试图的时候就放在demo文件夹即可
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(args.img, result,
                            palette=get_palette(args.palette),
                            show=False, opacity=args.opacity)
    mmcv.mkdir_or_exist(args.out)
    out_path = osp.join(args.out, osp.basename(args.img))
    cv2.imwrite(out_path, img)
    print(f"Result is save at {out_path}")
    '''        
        
    
    # 遍历文件夹
    # test_jpg_path = r'data/VOCdevkit/VOC2007/test_jpg'
    # test_jpg_path = r'data/test'
    test_jpg_path = r'/home/sylu/workspace/mjg/nelson/data/NG_Cropped'
    import os
    import numpy as np
    import time
    from tqdm import tqdm
    imgs = os.listdir(test_jpg_path)
    for img_path in tqdm(imgs):
        img_path = os.path.join(test_jpg_path,img_path)
        image = cv2.imread(img_path)  # 此处既然已经读取了原图，后续inference_model传入仅需将读取后的图像传给模型即可
        
        start = time.time()
        result = inference_model(model, image)
        end = time.time()
        print(f'推理时间： {end-start}秒')  # 以秒为单位  此处由于模型以及数据文件均较大，推理时间由14~18秒不等（后续稳定在14~15秒左右），显存占用： 9154MiB / 32768MiB
        '''
        5%|████▌                                                                                         | 20/410 [05:30<1:40:35, 15.48s/it]推理时间： 14.2584068775177秒
        Result is save at results_iter_80000/0607-1216_OK_srcTray_3_srcIndex_29_ACSYM_acqName_1_5c_2_Barcode_7cd304644e4947f5_0_0_test.jpg
        5%|████▊                                                                                         | 21/410 [05:45<1:40:09, 15.45s/it]推理时间： 14.250216960906982秒
        Result is save at results_iter_80000/15_huahenA-1_huahenA_srcTray_1_srcIndex_15_DL_result_0_0_1_HuaShang.jpg
        5%|█████                                                                                         | 22/410 [06:01<1:39:50, 15.44s/it]推理时间： 14.227401733398438秒
        Result is save at results_iter_80000/0222-1344_NG_srcTray_1_srcIndex_18_ACSYM_acqName_1_5c_2_0206_1_0_test_1_LouGuang.jpg
        6%|█████▎                                                                                        | 23/410 [06:16<1:39:35, 15.44s/it]推理时间： 14.248859405517578秒
        Result is save at results_iter_80000/0607-1216_OK_srcTray_1_srcIndex_50_ACSYM_acqName_1_5c_2_Barcode_7cd307644e4b672b_1_0_test.jpg
        ......
        10%|█████████▍                                                                                    | 41/410 [10:53<1:34:27, 15.36s/it]推理时间： 14.165016651153564秒
        Result is save at results_iter_80000/0523-0923_NG_srcTray_2_srcIndex_52_ACSYM_acqName_0_5c_2_Barcode_7cd31564546606fd_1_1_test_YiMo.jpg
        10%|█████████▋                                                                                    | 42/410 [11:08<1:34:00, 15.33s/it]推理时间： 14.204516649246216秒
        Result is save at results_iter_80000/0605-2053_OK_srcTray_14_srcIndex_32_ACSYM_acqName_0_5c_2_Barcode_7cd3046450959765_202_202_0_1_test.jpg
        10%|█████████▊                                                                                    | 43/410 [11:23<1:33:43, 15.32s/it]推理时间： 14.195263624191284秒
        Result is save at results_iter_80000/0605-2053_OK_srcTray_11_srcIndex_39_ACSYM_acqName_0_5c_2_Barcode_7cd30464509e692c_77_77_1_1_test.jpg
        '''
        
        '''
        补充——deeplabv3+速度：
        单卡推理速度如下：
        1%|█                                     | 1/122 [00:09<18:14,  9.04s/it]推理时间： 0.5354411602020264秒
        2%|███▎                                 | 2/122 [00:09<08:20,  4.17s/it]推理时间： 0.49387049674987793秒
        2%|████▉                                | 3/122 [00:10<05:06,  2.57s/it]推理时间： 0.5091474056243896秒
        3%|██████▌                               | 4/122 [00:11<03:39,  1.86s/it]推理时间： 0.470505952835083秒
        4%|████████▏                            | 5/122 [00:11<02:47,  1.43s/it]推理时间： 0.47536134719848633秒
        5%|█████████▊                           | 6/122 [00:12<02:15,  1.17s/it]推理时间： 0.46993589401245117秒
        6%|███████████▍                         | 7/122 [00:13<01:57,  1.02s/it]推理时间： 0.47842836380004883秒
        7%|█████████████                         | 8/122 [00:14<01:45,  1.08it/s]推理时间： 0.49158215522766113秒
        7%|██████████████▊                      | 9/122 [00:14<01:37,  1.16it/s]推理时间： 0.4747023582458496秒
        8%|████████████████▎                    | 10/122 [00:15<01:29,  1.26it/s]推理时间： 0.48772644996643066秒
        
        注：上述包含读取图像，图像与结果图融合等多个数据处理的时间，网络实际推理时间为：0.10~0.12秒左右
        <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        (3318, 1770, 3) (3318, 1770, 3) (3318, 3540, 3)
        17%|████████████████████████▏                 | 319/1922 [04:43<25:21,  1.05it/s]推理时间： 0.10497164726257324秒
        <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        (3318, 1771, 3) (3318, 1771, 3) (3318, 3542, 3)
        17%|████████████████████████▎                 | 320/1922 [04:43<24:59,  1.07it/s]推理时间： 0.10782861709594727秒
        <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        (3318, 1771, 3) (3318, 1771, 3) (3318, 3542, 3)
        17%|████████████████████████▍                 | 321/1922 [04:44<24:59,  1.07it/s]推理时间： 0.11038613319396973秒
        <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        (3334, 1780, 3) (3334, 1780, 3) (3334, 3560, 3)
        17%|████████████████████████▍                 | 322/1922 [04:45<25:06,  1.06it/s]推理时间： 0.10666728019714355秒
        <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        (3336, 1780, 3) (3336, 1780, 3) (3336, 3560, 3)
        17%|████████████████████████▌                  | 323/1922 [04:46<24:53,  1.07it/s]推理时间： 0.1121978759765625秒
        <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        (3330, 1777, 3) (3330, 1777, 3) (3330, 3554, 3)
        17%|████████████████████████▌                  | 324/1922 [04:47<24:42,  1.08it/s]推理时间： 0.11419177055358887秒
        <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        (3330, 1777, 3) (3330, 1777, 3) (3330, 3554, 3)
        17%|████████████████████████▋                 | 325/1922 [04:48<25:02,  1.06it/s]推理时间： 0.10715985298156738秒
        '''
                
        '''仅保存mask图像
        print(f'image类型：{type(image)}, result类型：{type(result[0])}')  # 返回的结果是list类型的，每个对象为numpy.ndarray图像的mask结果
        print(f'image形状：{image.shape}, result形状：{result[0].shape}')
        print(np.unique(result[0]))  # 统计像素值
        mmcv.mkdir_or_exist(args.out)
        out_path = osp.join(args.out, osp.basename(img_path))
        cv2.imwrite(out_path, result[0])
        print(f"Result is save at {out_path}")
        continue  # 仅保存mask结果图后续的内容不要了
        '''
        from mmseg.structures import SegDataSample
        from mmseg.visualization import SegLocalVisualizer
        seg_local_visualizer = SegLocalVisualizer(
            vis_backends=[dict(type='LocalVisBackend')],
            save_dir='./')
        seg_local_visualizer.dataset_meta = dict(
        classes=("_background_","BD_beng","lou_guang","jiao_beng","you_mo_yin","hua_shang","yi_mo"),
        palette=[(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128)])

        # 当`show=True`时，直接显示结果，
        # 当 `show=False`时，结果将保存在本地文件夹中。
        out_file = ''
        seg_local_visualizer.add_datasample(out_file, image,
                                            result, show=False,withLabels=False)
        
        # # 获得推理的mask图像
        # img_zero = np.zeros(image.shape)
        # pre_mask = model.show_result(img_zero, result,
        #                         palette=get_palette(args.palette),
        #                         show=False, opacity=args.opacity)
        
        # 直接对获得推理的mask图像进行染色得到可视化的预测结果（预测的mask为result[0]）（单张图像依次推理的情况下）
        colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128)]  # 每一类别的标签（这里保证mask与预测图颜色一致）
        orininal_h  = image.shape[0]
        orininal_w  = image.shape[1]
        pre_mask = np.reshape(np.array(colors, np.uint8)[np.reshape(result[0], [-1])], [orininal_h, orininal_w, -1])
        pre_mask = pre_mask[:, :, [2, 1, 0]]
        
        mmcv.mkdir_or_exist(args.out)
        out_path = osp.join(args.out, osp.basename(img_path))
        # cv2.imwrite(out_path, img)
        # print(f"Result is save at {out_path}")
        
        '''注：一与二选其一
        一、此处将原图与融合后的图像、结果图共三张图水平堆叠在一起（适用于没有mask的test图像）
        顺序如下：
        原图  结果图与原图融合后的图像 结果图
        '''
        """
        # 转换为 NumPy 数组
        np_image = np.array(image)  # 原图
        np_r_image = np.array(img)  # 结果图与原图融合后的图像 结果图
        
        stacked_image = np.hstack((np_image, np_r_image, pre_mask))

        # 保存图像为文件（例如JPEG格式）
        cv2.imwrite(out_path, stacked_image)
        print(f"Result is save at {out_path}")
        """
        
        '''
        此处增加该段代码是为了做模型对比，堆叠顺序为:结果图与原图融合后的图像 结果图（纵向堆叠）
        '''
        # 首先对返回内容进行了修改，加了判断条件以判断结果中是否存在NG缺陷
        pixel = np.unique(result[0])  # 统计预测结果的像素值
        NG = False  # 判断是否存在缺陷
        if pixel.any():  # any函数，任意一个元素不为0，输出为True（有缺陷）
            NG = True
        if NG:
            save_path = osp.join(args.out, 'NG', osp.basename(img_path))
        else:
            save_path = osp.join(args.out, 'OK', osp.basename(img_path))
                    
        # 转换为 NumPy 数组
        np_r_image = np.array(img)  # 结果图与原图融合后的图像
        
        stacked_image = np.vstack((np_r_image, pre_mask))

        # 保存图像为文件（例如JPEG格式）
        cv2.imwrite(save_path, stacked_image)
        print(f"Result is save at {save_path}")
        
        
        '''注：一与二选其一
        二、此处将原图/标签与结果图与原图融合后的图像/结果图堆叠在一起（适用于有mask的val/test图像）
        顺序如下：
        原图  结果图与原图融合后的图像
        标签  结果图
        '''
        
        """
        # 转换为 NumPy 数组
        np_image = np.array(image)  # 原图
        
        colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128)]  # 每一类别的标签（这里保证mask与预测图颜色一致）
        '''
        ["_background_","BD_beng","lou_guang","jiao_beng","you_mo_yin","hua_shang","yi_mo"]  # AC
        
        mask2former_beitv2_adapter_large_896_80k_ac_ms.py模型：
        [>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 511/511, 0.1 task/s, elapsed: 7313s, ETA:     0s2023-12-20 18:35:04,528 - mmseg - INFO - per class results:
        2023-12-20 18:35:04,529 - mmseg - INFO - 
        +--------------+-------+-------+
        |    Class     |  IoU  |  Acc  |
        +--------------+-------+-------+
        | _background_ | 99.86 | 99.91 |
        |   BD_beng    | 65.78 | 90.11 |
        |  lou_guang   | 77.76 | 90.65 |
        |  jiao_beng   | 84.96 | 94.35 |
        |  you_mo_yin  |  84.1 | 93.07 |
        |  hua_shang   | 58.19 |  85.3 |
        |    yi_mo     | 86.67 | 94.63 |
        +--------------+-------+-------+
        +-------+-------+-------+
        |  aAcc |  mIoU |  mAcc |
        +-------+-------+-------+
        | 99.86 | 79.62 | 92.57 |
        +-------+-------+-------+
        
        deeplabv3+模型：
        510 / 511: mIou-72.35%; mPA-85.24%; Accuracy-99.83%
        ===>_background_:       Iou-99.83; Recall (equal to the PA)-99.91; Precision-99.91
        ===>BD_beng:    Iou-59.49; Recall (equal to the PA)-73.74; Precision-75.47
        ===>lou_guang:  Iou-67.1; Recall (equal to the PA)-84.14; Precision-76.81
        ===>jiao_beng:  Iou-79.98; Recall (equal to the PA)-93.96; Precision-84.31
        ===>you_mo_yin: Iou-76.54; Recall (equal to the PA)-88.63; Precision-84.87
        ===>hua_shang:  Iou-46.84; Recall (equal to the PA)-71.38; Precision-57.67
        ===>yi_mo:      Iou-76.71; Recall (equal to the PA)-84.92; Precision-88.81
        ===> mIoU: 72.35; mPA: 85.24; Accuracy: 99.83
        
        综上所述：mask2former_beitv2_adapter_large_896_80k_ac_ms.py模型优于deeplabv3+模型，但速度上慢了大约30倍，完全失去了实时性（15s/img，即4imgs/min，即240imgs/h）
        但mask2former_beitv2_adapter_large_896_80k_ac_ms.py模型可用于非实时推理以取得更佳效果
        mask2former_beitv2_adapter_large_896_80k_ac_ms.py是896*896大小的输入图像，而deeplabv3+采用的是512*512大小的输入图像，由于原图像尺寸较大且不便裁剪因此较大的输入更有利（图像损失会减小很多）
        
        deeplabv3+混淆矩阵：
        +--------------+-------------+--------+--------+--------+-----------+-----------+---------+
        |              | _background_ | BD_beng|lou_guang|jiao_beng|you_mo_yin |hua_shang  | yi_mo|
        +--------------+-------------+--------+--------+--------+-----------+-----------+---------+
        | _background_ | 1378549533  | 138229 | 88825  | 98813  | 215873    | 177794    | 482243  |
        | BD_beng      | 169642      | 513994 | 8457   | 72     | 621       | 23        | 4235    |
        | lou_guang    | 58033       | 390    | 322410 | 0      | 0         | 208       | 2156    |
        | jiao_beng    | 31810       | 2225   | 42     | 531570 | 0         | 0         | 69      |
        | you_mo_yin   | 154625      | 0      | 0      | 0      | 1214412   | 464       | 668     |
        | hua_shang    | 97537       | 0      | 0      | 0      | 0         | 243215    | 0       |
        | yi_mo        | 663512      | 26185  | 0      | 52     | 0         | 0         | 3884233 |
        +--------------+-------------+--------+--------+--------+-----------+-----------+---------+

        '''
        # 注意此处需要外部给出mask_path的地址
        mask_path = r'data/VOCdevkit/VOC2007/test_mask'
        mask = os.path.join(mask_path, os.path.splitext(os.path.basename(img_path))[0]+'.png')
        mask       = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)  # 这里由于部分数据仍然是三通道的因此需要转为灰度图单通道的
        # （此处三个通道上数据是一致的，在训练时并没有产生额外的影响）
        orininal_h  = mask.shape[0]
        orininal_w  = mask.shape[1]
        mask = np.reshape(np.array(colors, np.uint8)[np.reshape(mask, [-1])], [orininal_h, orininal_w, -1])
        
        mask = mask[:, :, [2, 1, 0]]  # 重要，colors的定义时RGB格式的，此处是按照cv2读取的，读取的顺序是BGR格式，因此要进行transpose
        
        # 将原图像与mask图像竖着堆叠在一起
        stacked_image_orininal = np.vstack((np_image, mask))
        # 将原图像与结果图的混合图像与预测的结果图像竖着堆叠在一起
        np_r_image = np.vstack((img, pre_mask))
        
        # 检查两张图像的形状是否相同
        if stacked_image_orininal.shape[:2] == np_r_image.shape[:2]:
            # 将两张图像水平堆叠在一起
            stacked_image = np.hstack((stacked_image_orininal, np_r_image))
            # 保存图像为文件（例如JPEG格式）
            cv2.imwrite(out_path, stacked_image)
            print(f"Result is save at {out_path}")
        else:
            print("两张图像的形状不匹配，无法水平堆叠。")
        """
    
if __name__ == '__main__':
    main()