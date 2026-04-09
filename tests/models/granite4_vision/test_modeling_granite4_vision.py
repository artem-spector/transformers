# Copyright 2025 IBM. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch Granite4Vision model."""

import unittest

import pytest

from transformers import (
    CLIPVisionConfig,
    GraniteConfig,
    Granite4VisionConfig,
    Granite4VisionForConditionalGeneration,
    Granite4VisionModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


if is_vision_available():
    pass


class Granite4VisionModelTester(VLMModelTester):
    base_model_class = Granite4VisionModel
    config_class = Granite4VisionConfig
    conditional_generation_class = Granite4VisionForConditionalGeneration
    text_config_class = GraniteConfig
    vision_config_class = CLIPVisionConfig

    def __init__(self, parent, **kwargs):
        # Vision hidden_size must be divisible by 64 (QFormer num_attention_heads = hidden_size // 64)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("num_hidden_layers", 2)
        # Image/patch sizes: image_side = image_size // patch_size must be divisible by window_side
        kwargs.setdefault("image_size", 8)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("projection_dim", 64)
        kwargs.setdefault("num_patches_per_image", 2)
        # Granite4Vision-specific
        kwargs.setdefault("downsample_rate", "1/2")
        kwargs.setdefault("deepstack_layer_map", [[1, 0]])
        kwargs.setdefault("use_image_newline_parameter", True)
        kwargs.setdefault("use_spatial_sampling", False)
        kwargs.setdefault("projector_dropout", 0.0)
        kwargs.setdefault("image_token_index", kwargs.get("image_token_id", 3))

        # Compute num_image_tokens after downsampling:
        # image_side = image_size/patch_size = 4, ds 1/2 -> patches_h = patches_w = 2
        # pinpoints [[8,8]] -> scale 1x1 -> current_h = current_w = 2
        # unpadded = 2*2 = 4, newline = 2, base = 2*2 = 4 -> total = 10
        kwargs.setdefault("num_image_tokens", 10)

        super().__init__(parent, **kwargs)

    def create_pixel_values(self):
        """Granite4Vision expects 5D pixel_values: (batch_size, num_patches, channels, height, width)"""
        return floats_tensor(
            [
                self.batch_size,
                self.num_patches_per_image,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )

    def get_additional_inputs(self, config, input_ids, pixel_values):
        """Granite4Vision requires image_sizes tensor"""
        return {
            "image_sizes": torch.tensor([[self.image_size, self.image_size]] * self.batch_size),
        }

    def get_config(self):
        config = super().get_config()
        config.image_grid_pinpoints = [[self.image_size, self.image_size]]
        config.downsample_rate = self.downsample_rate
        config.deepstack_layer_map = self.deepstack_layer_map
        config.use_image_newline_parameter = self.use_image_newline_parameter
        config.use_spatial_sampling = self.use_spatial_sampling
        config.projector_dropout = self.projector_dropout
        return config


@require_torch
class Granite4VisionModelTest(VLMModelTest, unittest.TestCase):
    """
    Model tester for `Granite4VisionForConditionalGeneration`.
    """

    model_tester_class = Granite4VisionModelTester
    skip_test_image_features_output_shape = True
    test_torch_exportable = False
    # Custom layer-by-layer forward doesn't support output_attentions
    # (GraniteDecoderLayer discards attention weights internally)
    test_attention_outputs = False
    has_attentions = False

    # get_image_features returns deepstack (llm_layer, features) tuples, not ModelOutput
    @unittest.skip("get_image_features returns deepstack tuples, not ModelOutput")
    def test_get_image_features_output_0(self):
        pass

    @unittest.skip("get_image_features returns deepstack tuples, not ModelOutput")
    def test_get_image_features_output_1(self):
        pass

    @unittest.skip("get_image_features returns deepstack tuples, not ModelOutput")
    def test_get_image_features_output_2(self):
        pass

    @unittest.skip("get_image_features returns deepstack tuples, not ModelOutput")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip("get_image_features returns deepstack tuples, not ModelOutput")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip("Base model forward returns ModelOutputWithPast, not CausalLMOutput with loss")
    def test_training(self):
        pass

    @unittest.skip("QFormer submodules not initialized by init_weights from meta device")
    def test_can_init_all_missing_weights(self):
        pass

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Custom layer-by-layer forward has graph breaks incompatible with fullgraph compile")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("Blip2QFormerModel in WindowQFormerDownsampler does not support SDPA dispatch")
    def test_can_set_attention_dynamically_composite_model(self):
        pass
