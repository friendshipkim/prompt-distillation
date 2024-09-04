import logging
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from huggingface_hub import PyTorchModelHubMixin
from einops import rearrange

from model_utils import (
    EMBEDDING_TRANSFORM_STRATEGIES,
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
    mean_pool,
    extract_last_n,
    mean_sliding_window,
    rearrange_kv_tuple,
)

logger = logging.getLogger(__name__)


class PrefixEncoder(nn.Module):
    """Module that returns key-value pairs for prefix tokens."""

    prefix_projection: bool  # Whether to add an extra projection layer for prefix tokens
    prefix_len: int  # the number of prefix tokens for each layer
    model_dim: int  # Hidden dimension of backbone model
    output_dim: int  # Output dimension of prefix encoder (2 {key, value} * model_dim * num_layers)
    embedding: nn.Module  # Embedding layer for prefix tokens

    # only for prefix_projection
    bottleneck_dim: int  # Bottleneck dimension for prefix_encoder, default: model_dim
    transform: nn.Module  # Module that transforms prefix embeddings into key-value pairs
    dropout: nn.Module  # Dropout layer for prefix_encoder

    def __init__(
        self,
        prefix_projection: bool,
        prefix_len: int,
        model_dim: int,
        num_layers: int,
        # necessary for prefix_projection
        bottleneck_dim: int = None,
        drop_p: float = None,
    ) -> None:
        super().__init__()
        self.prefix_projection = prefix_projection
        self.prefix_len = prefix_len
        self.model_dim = model_dim
        self.output_dim = model_dim * 2 * num_layers

        if prefix_projection:
            self.bottleneck_dim = self.model_dim if bottleneck_dim is None else bottleneck_dim
            self.embedding = nn.Embedding(self.prefix_len, self.model_dim)
            self.transform = nn.Sequential(
                nn.Linear(self.model_dim, self.bottleneck_dim),
                nn.Tanh(),
                nn.Linear(self.bottleneck_dim, self.output_dim),
            )
            self.dropout = nn.Dropout(drop_p)
        else:
            self.embedding = nn.Embedding(self.prefix_len, self.output_dim)

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        if self.prefix_projection:
            prefix_embeds = self.embedding(prefix_tokens)
            prefix_embeds = self.transform(prefix_embeds)
            prefix_embeds = self.dropout(prefix_embeds)
        else:
            prefix_embeds = self.embedding(prefix_tokens)
        return prefix_embeds


class PatchEncoder(nn.Module):
    """Module that transforms teacher embeddings into key-value pairs for student model."""

    patch_projection: bool  # Whether to add an extra bottleneck layer for patch tokens
    patch_len: int  # the number of prefix patch for each layer
    teacher_dim: int  # Hidden dimension of the teacher model
    student_dim: int  # Hidden dimension of the student model
    output_dim: int  # Output dimension of patch encoder (2 {key, value} * student_dim * num_layers)
    transform: nn.Module  # Module that transforms patch embeddings into key-value pairs
    dropout: nn.Module  # Dropout layer

    # only for prefix_projection
    bottleneck_dim: int  # Bottleneck dimension for patch_encoder, default: student_dim

    def __init__(
        self,
        patch_projection: bool,
        # patch_len: int,
        teacher_dim: int,
        student_dim: int,
        student_num_layers: int,
        # necessary for patch_projection
        bottleneck_dim: int = None,
        drop_p: float = None,
    ) -> None:
        super().__init__()
        self.patch_projection = patch_projection
        # self.patch_len = patch_len
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.output_dim = student_dim * 2 * student_num_layers

        # old version
        # self.bottleneck_dim = (
        #     student.config.n_embd * 4 if bottleneck_dim is None else bottleneck_dim
        # )
        # self.embedding_transform = nn.Sequential(
        #     nn.Linear(self.teacher_dim, self.bottleneck_dim),
        #     nn.Dropout(self.teacher.config.attn_pdrop),
        #     nn.GELU(),  # TODO consider tanh or normalization here.
        #     nn.Linear(self.bottleneck_dim, encoder_hidden_dim * patch_len),
        # )
        # self.embedding_transform_kv = nn.Sequential(
        #     nn.Linear(self.teacher_dim, self.bottleneck_dim),
        #     nn.Dropout(
        #         self.teacher.config.attn_pdrop
        #     ),  # TODO make it compatible with non-gpt models
        #     nn.GELU(),  # TODO consider tanh or normalization here.
        #     nn.Linear(
        #         self.bottleneck_dim,
        #         encoder_hidden_dim * self.num_prefix_tokens * 2 * self.student.config.n_layer,
        #     ),
        # )

        if patch_projection:
            self.bottleneck_dim = self.student_dim if bottleneck_dim is None else bottleneck_dim
            self.transform = nn.Sequential(
                nn.Linear(self.teacher_dim, self.bottleneck_dim),
                nn.Tanh(),
                nn.Linear(self.bottleneck_dim, self.output_dim),
            )
        else:
            self.transform = nn.Sequential(
                nn.Linear(self.teacher_dim, self.output_dim),
                nn.Tanh(),
            )
        self.dropout = nn.Dropout(drop_p)

    def forward(self, teacher_embeds: torch.Tensor) -> torch.Tensor:
        # teacher_embeds: [batch_size, patch_len, teacher_dim]
        # output: [batch_size, patch_len, num_layers * 2, student_dim]
        patch_embeds = self.transform(teacher_embeds)
        patch_embeds = self.dropout(patch_embeds)
        patch_embeds = rearrange(patch_embeds, 'b p (l d) -> b p l d', d=self.student_dim)
        return patch_embeds


class PrefixModel(nn.Module, PyTorchModelHubMixin):
    """A class of model that prepends a prefix to the input sequence."""

    backbone: transformers.AutoModelForCausalLM
    backbone_lora: bool  # Whether to use LoRA for the backbone model
    backbone_tokenizer: transformers.PreTrainedTokenizer  # backbone's tokenizer

    hidden_size: int  # Hidden dimension of backbone model
    num_heads: int  # Number of attention heads in backbone model
    num_layers: int  # Number of layers in backbone model

    prefix_len: int  # the number of prefix tokens for each layer
    prefix_projection: bool  # Whether to add an extra projection layer for prefix tokens
    prefix_encoder: PrefixEncoder  # Module that returns key-value pairs for prefix tokens
    prefix_input_ids: torch.Tensor  # prefix input token ids (0, 1, 2, ..., prefix_len-1)
    prefix_attention_mask: torch.Tensor  # prefix attention mask (1, 1, 1, ..., 1)

    def __init__(
        self,
        backbone: transformers.AutoModelForCausalLM,
        tokenizer: transformers.PreTrainedTokenizer,
        prefix_len: int,
        prefix_projection: bool,
        backbone_dropout_disabled: bool = False,
        backbone_lora: bool = False,
        bottleneck_dim: int = None,
        drop_p: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        if backbone_lora:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_int8_training,
            )

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            print("Initializing LORA model with config:", peft_config)
            self.backbone = prepare_model_for_int8_training(self.backbone)
            self.backbone = get_peft_model(self.backbone, peft_config)
        ######################################################
        self.tokenizer = tokenizer
        self.isbloom = "bloom" in self.backbone.name_or_path
        self.hidden_size = backbone.config.hidden_size
        self.num_heads = backbone.config.num_attention_heads
        self.num_layers = backbone.config.n_layer
        self.prefix_len = prefix_len
        self.prefix_projection = prefix_projection

        # prefix encoder
        self.prefix_encoder = PrefixEncoder(
            prefix_projection=prefix_projection,
            prefix_len=prefix_len,
            model_dim=self.hidden_size,
            num_layers=self.num_layers,
            bottleneck_dim=bottleneck_dim,
            drop_p=drop_p,
        )

        # prefix input
        self.prefix_input_ids = torch.arange(self.prefix_len)
        self.prefix_attention_mask = torch.ones_like(self.prefix_input_ids)

        # disable dropout
        if backbone_dropout_disabled:
            print("Dropout disabled for backbone model")
            disable_dropout(self.backbone)

        # freeze backbone
        print("Freezing backbone model")
        freeze_params(self.backbone)

        # for generation
        self.backbone_prepare_inputs_for_generation = self.backbone.prepare_inputs_for_generation

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def get_prefix_attention_mask(self, batch_size: int):
        prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(self.device)
        return prefix_attention_mask

    def get_prefix(self, batch_size: int):
        prefix_tokens = self.prefix_input_ids.unsqueeze(0).expand(batch_size, -1).to(self.device)
        prefix_tokens = prefix_tokens[:, : self.prefix_len]
        past_key_values = self.prefix_encoder(prefix_tokens)
        # print("past_key_values", past_key_values.shape)

        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_len,
            self.num_layers * 2,
            self.num_heads,
            self.hidden_size // self.num_heads,
        )
        # gpt2
        #  - key: [batch_size, self.num_heads, kv_length, head_dim]
        #  - value: same as key
        # tuple of length num_layers, each with tensor of shape [2, <layerwise_shape>]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        if self.isbloom:
            # bloom
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            # tuple of length num_layers, each with tuple of length 2, each with tensor of shape [<layerwise_shape>]
            past_key_values = tuple(
                [
                    (
                        layer_past_key_values[0].flatten(0, 1).transpose(-1, -2),
                        layer_past_key_values[1].flatten(0, 1),
                    )
                    for layer_past_key_values in past_key_values
                ]
            )
        return past_key_values

    def generate(
        self,
        **kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # override backbone's prepare_inputs_for_generation
        self.backbone.prepare_inputs_for_generation = self.prepare_inputs_for_generation

        # generate
        outputs = self.backbone.generate(**kwargs)

        # restore backbone's prepare_inputs_for_generation
        self.backbone.prepare_inputs_for_generation = self.backbone_prepare_inputs_for_generation
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # prepare inputs for generation from the backbone (without past_key_values)
        model_kwargs = self.backbone_prepare_inputs_for_generation(*args, **kwargs)

        # offset position_ids by prefix_len
        model_kwargs["position_ids"] = model_kwargs["position_ids"] + self.prefix_len

        # first iter - if attention_mask is not None, concat prefix_attention_mask
        if model_kwargs.get("attention_mask", None) is not None:
            prefix_attention_mask = self.get_prefix_attention_mask(
                model_kwargs["input_ids"].shape[0]
            )
            model_kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
            )

        # first iter - if past_key_values is None, get prefix
        if model_kwargs["past_key_values"] is None:
            past_key_values = self.get_prefix(batch_size=model_kwargs["input_ids"].shape[0])
            model_kwargs["past_key_values"] = past_key_values

            # generate position_ids
            past_length = self.prefix_len
            input_ids = model_kwargs["input_ids"]
            attention_mask = model_kwargs["attention_mask"]
            position_ids = input_ids.new_zeros(input_ids.shape[0], input_ids.shape[1]).long()
            input_shape = input_ids.shape
            for i in range(input_ids.shape[0]):
                offset = attention_mask[i, past_length:].eq(0).long().cumsum(0)
                position_ids[i] = (
                    torch.arange(
                        past_length,
                        input_shape[-1] + past_length,
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    - offset
                )
            position_ids = position_ids.clamp(min=0)
            model_kwargs["position_ids"] = position_ids
        # TODO - embedding level generation
        # else:
        #     if model_kwargs["past_key_values"] is None:
        #         inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
        #         prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
        #         prompts = prompts.to(inputs_embeds.dtype)
        #         model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
        #         model_kwargs["input_ids"] = None

        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
            )
            kwargs["token_type_ids"] = None

        return model_kwargs

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        past_key_values = self.get_prefix(batch_size=input_ids.shape[0])

        prefix_attention_mask = self.get_prefix_attention_mask(batch_size=input_ids.shape[0])
        concat_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        outputs = self.backbone(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=concat_attention_mask,
            labels=labels,
            **kwargs,
        )

        return outputs


class PatchedModel(nn.Module, PyTorchModelHubMixin):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """

    teacher: transformers.AutoModelForCausalLM
    teacher_tokenizer: transformers.PreTrainedTokenizer
    student: transformers.AutoModelForCausalLM
    student_tokenizer: transformers.PreTrainedTokenizer
    teacher_lora: bool  # Whether to use LoRA for the teacher model
    student_lora: bool  # Whether to use LoRA for the student model
    student_num_layers: int  # Number of layers in student model
    student_num_heads: int  # Number of attention heads in student model

    patch_len: int  # The number of patch tokens prepended to the student inputs
    patch_encoder: nn.Module  # Module that transforms teacher embeddings into student past_key_values
    use_frozen_patch_as_input: bool  # Whether to train/evaluate on frozen embeddings
    # only set if patching level is task
    teacher_input: str  # patch prompt string shared by all inputs
    teacher_input_ids: torch.Tensor  # patch prompt encoded into token ids
    teacher_attention_mask: torch.Tensor  # patch prompt attention mask

    embedding_transform_strategy: str  # Way to transform teacher embedding into patch_len embeddings
    add_patch_tokens: bool  # Whether to add patch tokens to the teacher input sequence
    embeddings_from_layer_n: bool  # Which hidden state of teacher layer to use for embeddings, default: -1 (last)
    freeze_teacher: bool  # Whether to freeze teacher model, turn off dropout too
    freeze_student: bool  # Whether to freeze student model, turn off dropout too

    def __init__(
        self,
        teacher: transformers.AutoModelForCausalLM,
        teacher_tokenizer: transformers.PreTrainedTokenizer,
        student: transformers.AutoModelForCausalLM,
        student_tokenizer: transformers.PreTrainedTokenizer,
        patch_len: int,
        patch_projection: bool,
        freeze_strategy: str = "student",
        teacher_dropout_disabled: bool = False,
        student_dropout_disabled: bool = False,
        teacher_lora: bool = False,
        student_lora: bool = False,
        embedding_transform_strategy: str = "lastn",
        add_patch_tokens: bool = False,
        use_frozen_patch_as_input: bool = False,
        bottleneck_dim: int = None,
        drop_p: float = 0.0,
        embeddings_from_layer_n: int = None,
        random_patch: bool = False,
        ignore_prefix: bool = False,
        gate_init_val: float = 0,
        freeze_gate: bool = False,
        mlp: bool = False,
        offset_pos_id: bool = False,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.teacher_dim = teacher.config.hidden_size
        self.student_dim = student.config.hidden_size
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.gate = nn.Parameter(torch.zeros(1).fill_(gate_init_val))
        if freeze_gate:
            self.gate.requires_grad = False
        self.ignore_prefix = ignore_prefix
        self.offset_pos_id = offset_pos_id

        self.teacher_num_layers = self.teacher.config.n_layer
        self.student_num_layers = self.student.config.n_layer
        self.teacher_dtype = self.teacher.dtype
        self.student_dtype = self.student.dtype
        self.isbloom = "bloom" in self.student.name_or_path

        self.teacher_kv_dim = self.teacher.config.kv_dim
        self.student_kv_dim = self.student.config.kv_dim
        self.teacher_num_kv_heads = self.teacher.config.num_kv_heads
        self.student_num_kv_heads = self.student.config.num_kv_heads

        # LORA ==============================================
        if student_lora or teacher_lora:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_int8_training,
            )

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            if teacher_lora:
                print("Initializing LORA model with config:", peft_config)
                self.teacher = prepare_model_for_int8_training(self.teacher)
                self.teacher = get_peft_model(self.teacher, peft_config)
            if student_lora:
                print("Initializing LORA model with config:", peft_config)
                self.student = prepare_model_for_int8_training(self.student)
                self.student = get_peft_model(self.student, peft_config)
        # =================================================== #

        # patch ==============================================
        self.patch_len = patch_len
        self.bottleneck_dim = bottleneck_dim if patch_projection else self.student_dim
        # random patch embedding - similar to prefix + adapter
        self.random_patch = random_patch
        if self.random_patch:
            self.random_embeddings = nn.Parameter(
                torch.randn(self.student_num_layers, 2, self.patch_len, self.teacher_kv_dim)
            ).to(self.student_dtype)

        # TODO future use - get patch_embeds from a frozen model
        self.use_frozen_patch_as_input = use_frozen_patch_as_input
        if use_frozen_patch_as_input:
            # temp hack to set fixed sentence embedding size to 512.
            # TODO do this in a smarter way (figure it out from data? or make it an arg.)
            self.teacher_dim = 512

        # add patch tokens to teacher tokenizer
        # NOTE move this to main.py will affect reproducibility
        if add_patch_tokens:
            self.init_patch_tokens()

        # patch transform strategy
        assert embedding_transform_strategy in EMBEDDING_TRANSFORM_STRATEGIES
        self.embedding_transform_strategy = embedding_transform_strategy
        if embedding_transform_strategy == "last_n":
            self.embedding_proj = None
            self.patch_encoder = PatchEncoder(
                patch_projection=patch_projection,
                teacher_dim=self.teacher_dim,
                student_dim=self.student_kv_dim,
                student_num_layers=self.student_num_layers,
                bottleneck_dim=bottleneck_dim,
                drop_p=drop_p,
            )
        elif embedding_transform_strategy in ["last_and_project", "pool_and_project"]:
            self.embedding_proj = nn.Linear(self.teacher_dim, self.teacher_dim * self.patch_len)
            self.patch_encoder = PatchEncoder(
                patch_projection=patch_projection,
                teacher_dim=self.teacher_dim,
                student_dim=self.student_kv_dim,
                student_num_layers=self.student_num_layers,
                bottleneck_dim=bottleneck_dim,
                drop_p=drop_p,
            )
        elif embedding_transform_strategy in ["layerwise_pool_and_project", "layerwise_last_and_project"]:
            self.embedding_proj = nn.ModuleList(
                [
                    nn.Linear(self.teacher_dim, self.teacher_dim * self.patch_len)
                    for _ in range(self.student_num_layers)
                ]
            )
            self.patch_encoder = nn.ModuleList(
                [
                    PatchEncoder(
                        patch_projection=patch_projection,
                        teacher_dim=self.teacher_dim,
                        student_dim=self.student_kv_dim,
                        student_num_layers=1,
                        bottleneck_dim=bottleneck_dim,
                        drop_p=drop_p,
                    )
                    for _ in range(self.student_num_layers)
                ]
            )
        elif embedding_transform_strategy in ["layerwise_pool_and_share_project", "layerwise_last_and_share_project"]:
            self.embedding_proj = nn.Linear(self.teacher_dim, self.teacher_dim * self.patch_len)
            self.patch_encoder = nn.ModuleList(
                [
                    PatchEncoder(
                        patch_projection=patch_projection,
                        teacher_dim=self.teacher_dim,
                        student_dim=self.student_kv_dim,
                        student_num_layers=1,
                        bottleneck_dim=bottleneck_dim,
                        drop_p=drop_p,
                    )
                    for _ in range(self.student_num_layers)
                ]
            )
        elif embedding_transform_strategy in ["layerwise_share_pool_and_share_project", "layerwise_share_last_and_share_project"]:
            self.embedding_proj = nn.Linear(self.teacher_dim, self.teacher_dim * self.patch_len)
            self.patch_encoder = PatchEncoder(
                patch_projection=patch_projection,
                teacher_dim=self.teacher_dim,
                student_dim=self.student_kv_dim,
                student_num_layers=1,
                bottleneck_dim=bottleneck_dim,
                drop_p=drop_p,
            )
        elif "past_kv" in embedding_transform_strategy:
            if embedding_transform_strategy == "past_kv_copy":
                assert self.teacher_kv_dim == self.student_kv_dim
                self.embedding_proj = None
                self.patch_encoder = None
            elif embedding_transform_strategy == "past_kv_project_sharelayer_sharepatch":
                self.embedding_proj = nn.Linear(self.teacher_kv_dim, self.student_kv_dim)
                self.patch_encoder = None
            elif embedding_transform_strategy == "past_kv_project_sharelayer":
                self.embedding_proj = nn.ModuleList(
                    [
                        nn.Linear(self.teacher_kv_dim, self.student_kv_dim)
                        for _ in range(self.patch_len)
                    ]
                )
                self.patch_encoder = None
            elif embedding_transform_strategy == "past_kv_project_sharepatch":
                # teacher dim -> linear -> 4*teacher_dim -> ReLU -> linear -> student dim
                if mlp:
                    self.embedding_proj = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(self.teacher_kv_dim, 4 * self.teacher_kv_dim),
                                nn.ReLU(),
                                nn.Linear(4 * self.teacher_kv_dim, self.student_kv_dim),
                            )
                            for _ in range(self.student_num_layers * 2)
                        ]
                    )
                else:
                    self.embedding_proj = nn.ModuleList(
                        [
                            nn.Linear(self.teacher_kv_dim, self.student_kv_dim)
                            for _ in range(self.student_num_layers * 2)
                        ]
                    )
                self.patch_encoder = None
            elif embedding_transform_strategy == "past_kv_project_sharepatch_desc":
                # patch length is variable
                self.patch_len = -1
                self.embedding_proj = nn.ModuleList(
                    [
                        nn.Linear(self.teacher_kv_dim, self.student_kv_dim)
                        for _ in range(self.student_num_layers * 2)
                    ]
                )
                self.patch_encoder = None
            elif embedding_transform_strategy == "past_kv_sliding_window_project_sharepatch":
                self.embedding_proj = nn.ModuleList(
                    [
                        nn.Linear(self.teacher_kv_dim, self.student_kv_dim)
                        for _ in range(self.student_num_layers * 2)
                    ]
                )
                self.patch_encoder = None
        else:
            raise ValueError(f"unknown embedding transformation strategy {embedding_transform_strategy}")

        # match data type
        # or use float32 for everything
        if self.embedding_proj is not None:
            self.embedding_proj = self.embedding_proj.to(self.student_dtype)
        if self.patch_encoder is not None:
            self.patch_encoder = self.patch_encoder.to(self.student_dtype)

        self.embeddings_from_layer_n = (
            -1 if embeddings_from_layer_n is None else embeddings_from_layer_n
        )
        # =================================================== #

        # disable dropout
        if student_dropout_disabled:
            print("Dropout disabled for student model")
            disable_dropout(self.student)
        if teacher_dropout_disabled:
            print("Dropout disabled for teacher model")
            disable_dropout(self.teacher)

        # freeze models
        self.freeze(freeze_strategy)
        self.freeze_teacher = True if "teacher" in freeze_strategy else False
        self.no_grad_teacher = True if self.freeze_teacher and not add_patch_tokens else False
        self.freeze_student = True if "student" in freeze_strategy else False

        # for generation
        self.student_prepare_inputs_for_generation = self.student.prepare_inputs_for_generation

    def init_patch_tokens(self):
        print(f"Adding {self.patch_len} tokens to teacher tokenizer")
        patch_tokens = [f"<PATCH_{i}>" for i in range(self.patch_len)]
        self.teacher_tokenizer.add_tokens(patch_tokens, special_tokens=True)

        # initialize patch tokens to be the average of the original tokens
        original_input_embeddings = self.teacher.get_input_embeddings().weight.detach().clone()
        input_embeddings_avg = torch.stack([split.mean(0) for split in torch.tensor_split(original_input_embeddings, self.patch_len)])
        self.teacher.get_patch_embeddings().weight.data[:] = input_embeddings_avg[:]

        # adjust output embeddings to match the new input embeddings
        # original_output_embeddings = self.teacher.get_output_embeddings().weight.detach().clone()[: -self.patch_len]
        # output_embeddings_avg = torch.stack([split.mean(0) for split in torch.tensor_split(original_output_embeddings, self.patch_len)])
        # # output_embeddings_avg = original_output_embeddings.mean(dim=0, keepdim=True).expand([args.patch_len, -1])
        # self.teacher.get_output_embeddings().weight.data[-self.patch_len:] = output_embeddings_avg

        # not used for now
        # self.teacher.resize_token_embeddings(len(self.teacher_tokenizer))

    def _freeze_teacher(self):
        print("Freeze teacher model")
        freeze_params(self.teacher)

    def _freeze_student(self):
        print("Freeze student model")
        freeze_params(self.student)

    def freeze(self, freeze_strategy: str):
        assert freeze_strategy in FREEZE_STRATEGIES

        if freeze_strategy == "teacher":
            self._freeze_teacher()
        elif freeze_strategy == "student":
            self._freeze_student()
        elif freeze_strategy == "teacher_and_student":
            self._freeze_teacher()
            self._freeze_student()
        elif freeze_strategy == "none":
            pass
        else:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

    @property
    def device(self) -> torch.device:
        return next(self.teacher.parameters()).device

    def get_patch_attention_mask(self, batch_size: int):
        patch_attention_mask = torch.ones(batch_size, self.patch_len).to(self.device)
        if self.ignore_prefix:
            patch_attention_mask.fill_(0)
        return patch_attention_mask

    def _process_teacher_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        # # TODO support pooler output
        # if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
        #     return outputs.pooler_output
        # else:
        #     last_hidden_state = outputs.hidden_states[self.embeddings_from_layer_n]

        # use last hidden state
        # last_hidden_state shape: [batch_size, seq_len, hidden_size]
        # output embeddings shape: [batch_size, patch_len, hidden_size]
        if self.embedding_transform_strategy == "last_n":
            last_hidden_state = outputs.hidden_states[self.embeddings_from_layer_n].to(self.student_dtype)
            embeddings = extract_last_n(last_hidden_state, attention_mask, self.patch_len)
            # embeddings = last_hidden_state[:, -self.patch_len :, :]
        elif self.embedding_transform_strategy == "last_and_project":
            last_hidden_state = outputs.hidden_states[self.embeddings_from_layer_n].to(self.student_dtype)
            # last token embeddings: [batch_size, hidden_size]
            pooled_embeddings = extract_last_n(last_hidden_state, attention_mask, 1).squeeze(1)
            # project embedding and reshape to [batch_size, patch_len, hidden_size]
            embeddings = self.embedding_proj(pooled_embeddings)
            embeddings = embeddings.view(-1, self.patch_len, self.teacher_dim)
        elif self.embedding_transform_strategy == "pool_and_project":
            last_hidden_state = outputs.hidden_states[self.embeddings_from_layer_n].to(self.student_dtype)
            # pooled_embeddings: [batch_size, hidden_size]
            pooled_embeddings = mean_pool(last_hidden_state, attention_mask)  # or max_pool, stack_pool

            # project embedding and reshape to [batch_size, patch_len, hidden_size]
            embeddings = self.embedding_proj(pooled_embeddings)
            embeddings = embeddings.view(-1, self.patch_len, self.teacher_dim)

        # use hidden states of multiple layers
        # hidden_states shape: [batch_size, num_layers, seq_len, hidden_size]
        elif "layerwise" in self.embedding_transform_strategy:
            # get regular interval of layers, but include the last one
            extract_layers = [int((i * self.teacher_num_layers) / self.student_num_layers) for i in range(self.student_num_layers)]
            extract_layers = [l + 1 for l in extract_layers]
            assert extract_layers[-1] == len(outputs.hidden_states) - 1

            # extract layers from hidden state tuple
            hidden_states = tuple(outputs.hidden_states[l].to(self.student_dtype) for l in extract_layers)

            if self.embedding_transform_strategy == "layerwise_pool_and_project":
                pooled_embeddings = tuple(mean_pool(hidden_state, attention_mask) for hidden_state in hidden_states)
                embeddings = tuple(proj(embed).view(-1, self.patch_len, self.teacher_dim) for embed, proj in zip(pooled_embeddings, self.embedding_proj))
                # tuple of length num_layers, each with tensor of shape [batch_size, patch_len, hidden_size]
            elif self.embedding_transform_strategy == "layerwise_last_and_project":
                pooled_embeddings = tuple(extract_last_n(hidden_state, attention_mask, 1).squeeze(1) for hidden_state in hidden_states)
                embeddings = tuple(proj(embed).view(-1, self.patch_len, self.teacher_dim) for embed, proj in zip(pooled_embeddings, self.embedding_proj))
                # tuple of length num_layers, each with tensor of shape [batch_size, patch_len, hidden_size]
            elif self.embedding_transform_strategy == "layerwise_pool_and_share_project":
                pooled_embeddings = tuple(mean_pool(hidden_state, attention_mask) for hidden_state in hidden_states)
                embeddings = tuple(self.embedding_proj(embed).view(-1, self.patch_len, self.teacher_dim) for embed in pooled_embeddings)
                # tuple of length num_layers, each with tensor of shape [batch_size, patch_len, hidden_size]
            elif self.embedding_transform_strategy == "layerwise_share_pool_and_share_project":
                pooled_embeddings = tuple(mean_pool(hidden_state, attention_mask) for hidden_state in hidden_states)
                embeddings = tuple(self.embedding_proj(embed).view(-1, self.patch_len, self.teacher_dim) for embed in pooled_embeddings)
                # tuple of length num_layers, each with tensor of shape [batch_size, patch_len, hidden_size]
            elif self.embedding_transform_strategy == "layerwise_last_and_share_project":
                pooled_embeddings = tuple(extract_last_n(hidden_state, attention_mask, 1).squeeze(1) for hidden_state in hidden_states)
                embeddings = tuple(self.embedding_proj(embed).view(-1, self.patch_len, self.teacher_dim) for embed in pooled_embeddings)
                # tuple of length num_layers, each with tensor of shape [batch_size, patch_len, hidden_size]

        # use past_key_values of teacher output
        elif "past_kv" in self.embedding_transform_strategy:
            bsz = attention_mask.shape[0]
            # tuple of length num_layers, each with tuple of length 2
            # each with tensor of shape [batch_size, num_heads, seq_len, teacher_dim // num_heads]
            teacher_kv = outputs["past_key_values"]

            # NOTE or plus 1 to incliude the last layer
            extract_layers = [int((i * self.teacher_num_layers) / self.student_num_layers) for i in range(self.student_num_layers)]
            # print(f"extract_layers: {extract_layers}")

            if "sliding_window" in self.embedding_transform_strategy:
                # stack past_key_values of extracted layers
                embeddings = torch.stack(tuple(rearrange_kv_tuple(teacher_kv[l], mean_sliding_window, attention_mask, self.patch_len) for l in extract_layers), dim=0).to(self.student_dtype)
            elif "desc" in self.embedding_transform_strategy:
                # stack past_key_values of extracted layers and extract content
                embeddings = torch.stack(tuple(rearrange_kv_tuple(teacher_kv[l], None, attention_mask, self.patch_len) for l in extract_layers), dim=0).to(self.student_dtype)
                self.patch_len = embeddings.shape[3]
            else:
                # stack past_key_values of extracted layers and extract last patch_len tokens
                embeddings = torch.stack(tuple(rearrange_kv_tuple(teacher_kv[l], extract_last_n, attention_mask, self.patch_len) for l in extract_layers), dim=0).to(self.student_dtype)
            assert (self.student_num_layers, 2, bsz, self.patch_len, self.teacher_kv_dim) == embeddings.shape

            if self.random_patch:
                embeddings = self.random_embeddings.unsqueeze(2).expand_as(embeddings)
            # rearrange it to match the shape of patch_encoder's input
            # [num_layers * 2, bsz, patch_len, teacher_dim]
            embeddings = rearrange(embeddings, 'l kv ... -> (l kv) ...')
            # [bsz, seq_len, num_layers * 2, teacher_dim]
            embeddings = rearrange(embeddings, 'l b p d -> b p l d')

            # # normalize by patch_len
            # embeddings = embeddings / self.patch_len

            if self.embedding_transform_strategy == "past_kv_copy":
                pass
            elif self.embedding_transform_strategy == "past_kv_project_sharelayer_sharepatch":
                embeddings = rearrange(embeddings, 'b p l d -> (b p l) d')
                embeddings = rearrange(self.embedding_proj(embeddings), '(b p l) d -> b p l d', b=bsz, p=self.patch_len, l=self.student_num_layers * 2)
            elif self.embedding_transform_strategy == "past_kv_project_sharelayer":
                embeddings = rearrange(embeddings, 'b p l d -> p (b l) d')
                embeddings = torch.stack(tuple(self.embedding_proj[i](embeddings[i]) for i in range(self.patch_len)), dim=0)
                embeddings = rearrange(embeddings, 'p (b l) d -> b p l d', b=bsz)
            elif self.embedding_transform_strategy == "past_kv_project_sharepatch":
                embeddings = rearrange(embeddings, 'b p l d -> l (b p) d')
                embeddings = torch.stack(tuple(self.embedding_proj[i](embeddings[i]) for i in range(self.student_num_layers * 2)), dim=0)
                embeddings = rearrange(embeddings, 'l (b p) d -> b p l d', b=bsz)
            elif self.embedding_transform_strategy == "past_kv_project_sharepatch_desc":
                embeddings = rearrange(embeddings, 'b p l d -> l (b p) d')
                embeddings = torch.stack(tuple(self.embedding_proj[i](embeddings[i]) for i in range(self.student_num_layers * 2)), dim=0)
                embeddings = rearrange(embeddings, 'l (b p) d -> b p l d', b=bsz)
            elif self.embedding_transform_strategy == "past_kv_sliding_window_project_sharepatch":
                embeddings = rearrange(embeddings, 'b p l d -> l (b p) d')
                embeddings = torch.stack(tuple(self.embedding_proj[i](embeddings[i]) for i in range(self.student_num_layers * 2)), dim=0)
                embeddings = rearrange(embeddings, 'l (b p) d -> b p l d', b=bsz)
            else:
                raise ValueError(f"unknown embedding transformation strategy {self.embedding_transform_strategy}")
        else:
            raise ValueError(
                f"unknown embedding transformation strategy {self.embedding_transform_strategy}"
            )
        return embeddings

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        if self.freeze_teacher:
            self.teacher.eval()
        model_output = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            # output_attentions=True
        )
        # teacher_attentions = model_output["attentions"]
        embeddings = self._process_teacher_output(model_output, attention_mask)
        return embeddings

    def embed_and_project(
        self,
        teacher_input_ids: Optional[torch.Tensor],
        teacher_attention_mask: Optional[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # check if teacher_input_ids is 2D tensor
        assert teacher_input_ids.dim() == 2

        # expand to beam size for generation
        if teacher_input_ids.shape[0] != batch_size:
            # given teacher_input_ids / attention masks of shape [bsz, patch_len]
            # expand them to [bsz * beam, patch_len] by repeating each row 'beam_size' times together
            beam_size = batch_size // teacher_input_ids.shape[0]
            teacher_input_ids = teacher_input_ids[:, None, :].repeat(1, beam_size, 1)
            teacher_input_ids = teacher_input_ids.view(batch_size, -1)

            teacher_attention_mask = teacher_attention_mask[:, None, :].repeat(
                1, beam_size, 1
            )
            teacher_attention_mask = teacher_attention_mask.view(batch_size, -1)

        if self.no_grad_teacher:
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                )
        else:
            embeddings = self.call_embedding_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            )

        # ====== transform hidden states
        if self.patch_encoder is None:
            past_key_values = embeddings
        elif isinstance(self.patch_encoder, PatchEncoder):
            # share patch encoder across layers
            if isinstance(embeddings, tuple):
                past_key_values = tuple(self.patch_encoder(embed) for embed in embeddings)
                past_key_values = torch.stack(past_key_values, dim=2)
                past_key_values = rearrange(past_key_values, 'b p l2 kv d -> b p (l2 kv) d')
            else:
                past_key_values = self.patch_encoder(embeddings)
        elif isinstance(self.patch_encoder, nn.ModuleList):
            past_key_values = tuple(encoder(embed) for embed, encoder in zip(embeddings, self.patch_encoder))
            past_key_values = torch.stack(past_key_values, dim=2)
            past_key_values = rearrange(past_key_values, 'b p l2 kv d -> b p (l2 kv) d')
        else:
            raise ValueError(f"unknown patch encoder {self.patch_encoder}")

        # breakpoint()
        assert past_key_values.shape == (batch_size, self.patch_len, self.student_num_layers * 2, self.student_kv_dim)
        past_key_values = rearrange(past_key_values, 'b p l (h d) -> b p l h d', h=self.student_num_kv_heads)
        # past_key_values = past_key_values.view(
        #     batch_size,
        #     self.patch_len,
        #     self.student_num_layers * 2,
        #     self.student_num_kv_heads,
        #     self.student_kv_dim // self.student_num_kv_heads,
        # )

        # gpt2
        #  - key: [batch_size, self.num_heads, kv_length, head_dim]
        #  - value: same as key
        # tuple of length num_layers, each with tensor of shape [2, <layerwise_shape>]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        if self.isbloom:
            # bloom
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            # tuple of length num_layers, each with tuple of length 2, each with tensor of shape [<layerwise_shape>]
            past_key_values = tuple(
                [
                    (
                        layer_past_key_values[0].flatten(0, 1).transpose(-1, -2),
                        layer_past_key_values[1].flatten(0, 1),
                    )
                    for layer_past_key_values in past_key_values
                ]
            )
        return past_key_values

    def generate(
        self,
        **kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # NOTE does not support batch generation
        # this is same as prefix model's generate
        # override student's prepare_inputs_for_generation
        self.student.prepare_inputs_for_generation = self.prepare_inputs_for_generation

        # HACK: cannot pass unknown kwargs to generate, save them as class attributes
        self.teacher_input_ids = kwargs.pop("teacher_input_ids")
        self.teacher_attention_mask = kwargs.pop("teacher_attention_mask")

        # generate from student
        if self.freeze_student:
            self.student.eval()
        outputs = self.student.generate(**kwargs)

        # restore student's prepare_inputs_for_generation
        self.student.prepare_inputs_for_generation = self.student_prepare_inputs_for_generation
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)

        is_first_iter = past_key_values is None
        # Omit tokens covered by past_key_values
        if is_first_iter:
            # no past_key_values, first step
            # print('generate prefix from teacher')
            past_key_values = self.embed_and_project(
                teacher_input_ids=self.teacher_input_ids,
                teacher_attention_mask=self.teacher_attention_mask,
                batch_size=input_ids.shape[0],
            )
        else:
            # print('pkv given')
            past_length = past_key_values[0][0].shape[2]
            # print("past_length", past_length)

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            # extract generations only
            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # attention mask is always given
        attention_mask = kwargs.get("attention_mask", None)
        # position id is always none
        position_ids = kwargs.get("position_ids", None)

        if is_first_iter:
            # print("generate first position ids, attention mask, gated attention mask")
            # generate the first postion ids
            masks = self.generate_masks(input_ids, self.teacher_attention_mask, attention_mask)
            position_ids = masks["position_ids"]
            attention_mask = masks["attention_mask"]
            gate_attention_mask = masks["gate_attention_mask"]
        else:
            # print("create position_ids on the fly for batch generation")
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

            # print("create gate_attention_mask on the fly")
            masks = self.generate_masks(input_ids, self.teacher_attention_mask, attention_mask)
            # apply offset to position_ids
            position_ids = position_ids + masks["position_ids"]
            attention_mask = masks["attention_mask"]
            gate_attention_mask = masks["gate_attention_mask"]

        # print("input_ids", input_ids.shape)
        # print("position_ids", position_ids.shape, position_ids)
        # print("attention_mask", attention_mask.shape)
        # print("gate_attention_mask", gate_attention_mask.shape)

        # breakpoint()
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "gate_attention_mask": gate_attention_mask,
            "token_type_ids": token_type_ids,
        }
        return model_inputs

    def prepare_inputs_for_generation_old(self, *args, **kwargs):
        # prepare inputs for generation from the student (without past_key_values)
        model_kwargs = self.student_prepare_inputs_for_generation(*args, **kwargs)

        # offset position_ids by prefix_len
        model_kwargs["position_ids"] = model_kwargs["position_ids"] + self.patch_len

        # first iter - if attention_mask is not None, concat prefix_attention_mask
        if model_kwargs.get("attention_mask", None) is not None:
            prefix_attention_mask = self.get_patch_attention_mask(
                model_kwargs["input_ids"].shape[0]
            )
            model_kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
            )

        # first iter - if past_key_values is None, get prefix
        if model_kwargs["past_key_values"] is None:
            past_key_values = self.embed_and_project(
                teacher_input_ids=self.teacher_input_ids,
                teacher_attention_mask=self.teacher_attention_mask,
                batch_size=model_kwargs["input_ids"].shape[0],
            )
            model_kwargs["past_key_values"] = past_key_values

            # # generate position_ids
            # past_length = self.patch_len
            # input_ids = model_kwargs["input_ids"]
            # attention_mask = model_kwargs["attention_mask"]
            # position_ids = input_ids.new_zeros(input_ids.shape[0], input_ids.shape[1]).long()
            # input_shape = input_ids.shape
            # for i in range(input_ids.shape[0]):
            #     offset = attention_mask[i, past_length:].eq(0).long().cumsum(0)
            #     position_ids[i] = (
            #         torch.arange(
            #             past_length,
            #             input_shape[-1] + past_length,
            #             dtype=torch.long,
            #             device=input_ids.device,
            #         )
            #         - offset
            #     )
            # position_ids = position_ids.clamp(min=0)
            # model_kwargs["position_ids"] = position_ids

        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
            )
            kwargs["token_type_ids"] = None

        return model_kwargs

    def generate_masks(self, input_ids, tr_attention_mask, st_attention_mask):
        # prefix attention mask, shape: [bsz, patch_len]
        if "desc" in self.embedding_transform_strategy:
            # variable kv length, use teacher attention mask
            prefix_attention_mask = tr_attention_mask
        else:
            prefix_attention_mask = self.get_patch_attention_mask(batch_size=input_ids.shape[0])
        prefix_attention_mask = prefix_attention_mask.expand(input_ids.shape[0], -1)
        concat_attention_mask = torch.cat([prefix_attention_mask, st_attention_mask], dim=1)
        gate_attention_mask = torch.cat([self.gate * prefix_attention_mask, st_attention_mask], dim=1)

        # starting from 0, shape: [1, student_seq_len]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)
        if self.offset_pos_id:
            # offset position id starts from the sum of prefix_attention_mask
            pos_start_idx = torch.zeros(input_ids.shape[0]) if self.ignore_prefix else prefix_attention_mask.sum(-1)
            # expand position id (1, student_seq_len) to (bsz, student_seq_len)
            position_ids = position_ids.expand(input_ids.shape[0], -1)
            position_ids = position_ids + pos_start_idx.unsqueeze(-1)
        return {
            "position_ids": position_ids.to(torch.long),
            "attention_mask": concat_attention_mask,
            "gate_attention_mask": gate_attention_mask,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        teacher_input_ids: Optional[torch.Tensor],
        teacher_attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        past_key_values = self.embed_and_project(
            teacher_input_ids=teacher_input_ids,
            teacher_attention_mask=teacher_attention_mask,
            batch_size=input_ids.shape[0],
        )
        # past_key_values = [item*self.gate for item in past_key_values]

        masks = self.generate_masks(input_ids, teacher_attention_mask, attention_mask)
        if self.freeze_student:
            self.student.eval()
        outputs = self.student(
            input_ids=input_ids,
            position_ids=masks["position_ids"],
            past_key_values=past_key_values,
            attention_mask=masks["attention_mask"],
            gate_attention_mask=masks["gate_attention_mask"],
            labels=labels,
            **kwargs,
        )

        # unpadded loss
        # logits = outputs.logits
        # # shift labels by one token
        # shifted_labels = labels[..., 1:].contiguous()
        # shifted_logits = logits[..., :-1, :].contiguous()
        # # print(f"shifted_labels: {shifted_labels.shape}")
        # # print(f"shifted_logits: {shifted_logits.shape}")
        # assert shifted_labels.size(1) == shifted_logits.size(1)

        # # mask of non-padding tokens
        # is_valid_token = shifted_labels >= 0

        # shifted_labels[~is_valid_token] = 0
        # log_probs = shifted_logits.log_softmax(-1).gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
        # loss = -log_probs[is_valid_token].mean()
        # outputs.loss = loss

        return outputs