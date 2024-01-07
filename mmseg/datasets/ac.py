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
    # 按照如下方式定义类别等信息（此时num_classes=7）
    # METAINFO = dict(
    #     classes=("_background_","BD_beng","lou_guang","jiao_beng","you_mo_yin","hua_shang","yi_mo"),
    #     palette=[[0], [1], [2], [3],
    #             [4], [5],[6]]
    # )

    
    # def __init__(self,
    #             img_suffix='.jpg',
    #             seg_map_suffix='.png',
    #             **kwargs) -> None:
    #     super().__init__(
    #         img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
    
    # 将上述内容修改为5类缺陷（此时num_classes=6）：
    METAINFO = dict(
        classes=("_background_","BD_beng","lou_guang", "you_mo_yin","hua_shang","yi_mo"),
        palette=[[0], [1], [2], [3],
                [4], [5]]
    )
    def __init__(self,
                img_suffix='.bmp',  # 后缀格式改为bmp
                seg_map_suffix='.png',
                **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
    