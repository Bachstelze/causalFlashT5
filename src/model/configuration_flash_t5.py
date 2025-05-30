import sys
from collections import OrderedDict
from typing import Mapping
import logging

from transformers import T5Config

AUTO_MAP = {
    "AutoModel": "modeling_flash_t5.FlashT5EncoderModel",
    "AutoModelForSeq2SeqLM": "modeling_flash_t5.FlashT5ForConditionalGeneration",
    "AutoModelForTokenClassification": "custom_heads_flash_t5.FlashT5ForTokenClassification",
    "AutoModelForQuestionAnswering": "custom_heads_flash_t5.FlashT5ForQuestionAnswering",
    "AutoModelForSequenceClassification": "custom_heads_flash_t5.FlashT5ForSequenceClassification",
}

class FlashT5Config(T5Config):

    model_type = "flash_t5"

    def __init__(
        self,
        decoder_start_token_id=0,
        pad_token_id=-100,
        use_glu_mlp=False,
        position_encoding_type="t5",
        use_randomized_position_encoding=False,
        label_smoothing=0.0,
        z_loss=None,
        attention_type="ref",
        max_sequence_length=1024,
        attention_dropout_rate=0.0,
        alibi_mode="symetric",
        use_triton_layernorm=False,
        use_triton_crossentropy=False,
        crossentropy_inplace_backward=False,
        use_gelu_act=True,
        use_full_bias_size=False,
        rotary_emb_fraction=1.0,
        rotary_base=10000,
        rotary_interleaved=False,
        rotary_scale_base=None,
        fire_mlp_width=32,
        use_masking=False,
        attention_scale=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.use_glu_mlp = use_glu_mlp
        self.position_encoding_type = position_encoding_type
        self.use_randomized_position_encoding = use_randomized_position_encoding
        self.label_smoothing = label_smoothing
        self.z_loss = z_loss
        self.attention_type = attention_type
        self.max_sequence_length = max_sequence_length
        self.alibi_mode = alibi_mode
        self.attention_dropout_rate = attention_dropout_rate
        self.use_triton_layernorm = use_triton_layernorm
        self.use_triton_crossentropy = use_triton_crossentropy
        self.crossentropy_inplace_backward = crossentropy_inplace_backward
        self.use_gelu_act = use_gelu_act
        self.use_full_bias_size = use_full_bias_size
        self.rotary_base = rotary_base
        self.rotary_interleaved = rotary_interleaved
        self.rotary_scale_base = rotary_scale_base
        self.rotary_emb_fraction = rotary_emb_fraction
        self.fire_mlp_width = fire_mlp_width
        self.use_masking = use_masking
        self.attention_scale = attention_scale

        self.auto_map = AUTO_MAP

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# Register model in Auto API
try:
    FlashT5Config.register_for_auto_class()
    for key, value in AUTO_MAP.items():
        str_to_class(value.split(".")[-1]).register_for_auto_class(key)
except:
    logging.warn("AutoRegister isn't available.")
