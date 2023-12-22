# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module(force=True)
class AcDataset(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
  
    # CLASSES = ("_background_","BD_beng","lou_guang","jiao_beng","you_mo_yin","hua_shang","yi_mo")  # AC
    CLASSES = ("BD_beng","lou_guang","jiao_beng","you_mo_yin","hua_shang","yi_mo")  # AC

    # PALETTE = [[0], [1], [2], [3],
    #            [4], [5],[6]]
    
    PALETTE = [[1], [2], [3],
               [4], [5],[6]]
    
    # PALETTE = [[0,0,0], [1,1,1], [2,2,2], [3,3,3],
    #            [4,4,4], [5,5,5],[6,6,6]]  # 调色板仅用来在可视化等处使用，训练的时候没有影响
    
    # PALETTE = [[0, 0, 0], [0, 192, 0], [196, 196, 196], [190, 153, 153],
    #            [180, 165, 180], [90, 120, 150], [
    #                102, 102, 156]]
               
    # def __init__(self, **kwargs):
    #     super(AcDataset, self).__init__(
    #         img_suffix='.jpg',
    #         seg_map_suffix='.png',
    #         reduce_zero_label=False,  ### 此处后面的有错误   # 当第一类为背景等需要忽略时，设置为True即可  # 参考链接：https://blog.csdn.net/qq_39967751/article/details/126272578
    #         **kwargs)
    
    def __init__(self,
                img_suffix='.jpg',
                seg_map_suffix='.png',
                **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)