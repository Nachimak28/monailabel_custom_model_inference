from typing import Callable, Sequence

import numpy as np
from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    Activationsd,
    CastToTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    ToNumpyd
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import BoundingBoxd, Restored

class LungLesionSeg(InferTask):
    """
    This provides the inference engine for our trained lung lesion segmentation model over the grand challenge validation dataset
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels="lesion",
        dimension=1,
        description="lung lesion segmentation model over the grand challenge validation dataset",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )


    def pre_transforms(self, data=None) -> Sequence[Callable]:

        keys = ("image",)
        return [
            LoadImaged(keys),
            AddChanneld(keys),
            Orientationd(keys, axcodes="LPS"),
            Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
            ScaleIntensityRanged(keys, a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
            CastToTyped(keys, dtype=np.float32), 
            EnsureTyped(keys)
        ]


    def inferer(self, data=None) -> Inferer:
        patch_size = (192, 192, 16)
        sw_batch_size, overlap = 1, 0.5
        inferer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode="gaussian",
            padding_mode="replicate",
        )
        return inferer
    
    def post_transforms(self, data):
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]