import os
import sys
import transformers
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, NewType
from transformers import HfArgumentParser
from alignment import SFTConfig

DataClassType = NewType("DataClassType", Any)


@dataclass
class SFTDistillConfig(SFTConfig):
    """
    Arguments related to the distillation process.
    """
    stduent_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the previous distilled model in this progressive distillation."},
    )
    patch_len: Optional[int] = field(
        default=100,
        metadata={"help": "Patch length"},
    )
    # patch_projection: bool = field(
    #     default=False,
    #     metadata={"help": "Patch projection"},
    # )
    embedding_transform_strategy: str = field(
        default="select_layer_all",
        metadata={"help": "Embedding transform strategy"},
    )
    embeddings_from_layer_n: List[int] = field(
        default=None,
        metadata={"help": "Embeddings from layer n"},
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Debug mode"},
    )
