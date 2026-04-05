from typing import Optional
import logging
from ..llava_next.configuration_llava_next import LlavaNextConfig

logger = logging.getLogger(__name__)

__all__ = ["Granite4VisionConfig"]


class Granite4VisionConfig(LlavaNextConfig):
    model_type = "granite4_vision"
    def __init__(
        self,
        downsample_rate=None,
        use_image_newline_parameter=True,
        deepstack_layer_map: Optional[list] = None,
        use_spatial_sampling: bool = False,
        spatial_stride: int = 2,
        spatial_vision_layer: int = -1,
        spatial_target_layers: Optional[list] = None,
        projector_dropout=0.1,
        **kwargs
    ):
        self.downsample_rate = downsample_rate
        self.use_image_newline_parameter = use_image_newline_parameter
        self.projector_dropout = projector_dropout

        # Deepstack layer map: list of (vision_layer_idx, llm_layer_idx) tuples.
        # Features from each vision layer are extracted, downsampled, and injected
        # at the corresponding LLM layer during forward pass.
        # e.g., [(-25, 12), (-17, 8), (-9, 4), (-1, 0)]
        if deepstack_layer_map is not None:
            self.deepstack_layer_map = [(int(v), int(l)) for v, l in deepstack_layer_map]
            assert len(self.deepstack_layer_map) == len(set(self.deepstack_layer_map)), "expecting no duplicates"
        else:
            self.deepstack_layer_map = None

        # Spatial sampling: extracts 4 groups from a single vision layer using
        # spatial offset sampling (top-left, top-right, bottom-left, bottom-right
        # of each 2x2 block), each injected at a different LLM layer.
        self.use_spatial_sampling = use_spatial_sampling
        self.spatial_stride = spatial_stride
        self.spatial_vision_layer = spatial_vision_layer
        self.spatial_target_layers = spatial_target_layers or [0, 10, 20, 30]

        super().__init__(**kwargs)
