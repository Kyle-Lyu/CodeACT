import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    default_data_collator,
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model

from utils.monitor import MemoryUsageCallback
from utils.dataset import DataCollatorWithDynamicPad, DataCollatorWithDynamicPack, create_dataset


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    low_cpu_mem_usage: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )

    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )

    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_nested_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Compute dtype for 4bit base models",
            "choices": ["float32", "float16", "bfloat16", "uint8"],
        },
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Quantization storage dtype for 4bit base models",
            "choices": ["uint8", "float32", "float16", "bfloat16"],
        },
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={
            "help": "Quantization type fp4 or nf4",
            "choices": ["fp4", "nf4"],
        },
    )
    

@dataclass
class DataTrainingArguments:
    data_path: str = field(
        metadata={"help": "Path to the json training data."}
    )
    model_type: str = field(
        metadata={
            "help": "Model type for preprocess data using specific tokenizer",
            "choices": ["codellama", "codegemma", "deepseek", "starcoder2", "codeqwen", "llama3"],
        }
    )
    max_seq_len: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum lenght of training samples. If None, use model.config.max_position_embeddings"}
    )
    pad_mode: Optional[str] = field(
        default="dynamic_pad", 
        metadata={
            "help": "Padding mode to preprocess the data.",
            "choices": ["dynamic_pad", "dynamic_pack"],
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training parameters {training_args}")

    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )
    if not data_args.max_seq_len:
        data_args.max_seq_len = getattr(config, "max_position_embeddings", None)
    if not data_args.max_seq_len:
        raise ValueError("Please specify the maximum training sample length, for example --max_seq_len 2048")
    logger.info(f"Data parameters {data_args}")

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    # set quantization config
    bnb_config = None
    if model_args.use_4bit_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
            bnb_4bit_quant_storage=compute_dtype,
        )

        if compute_dtype == torch.float16 and model_args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif model_args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=model_args.use_8bit_quantization)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
    )
    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        # model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    if model_args.use_4bit_quantization:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    
    if model_args.use_peft_lora:
        target_modules = model_args.lora_target_modules.split(",") if model_args.lora_target_modules != "all-linear" else model_args.lora_target_modules
        # more details at https://huggingface.co/docs/peft/v0.11.0/en/package_reference/lora#peft.LoraConfig
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        if training_args.local_rank == 0:
            print("=" * 80)
            model.print_trainable_parameters()
            print("=" * 80)
    else:
        all_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info("=" * 80)
        logger.info(f"trainable params: {all_params:,} || all params: {all_params:,} || trainable%: 100")
        logger.info("=" * 80)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    if data_args.model_type == "starcoder2":
        tokenizer.chat_template = "{{bos_token}}{{'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n'}}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n        {{ raise_exception('System messages are not allowed in this template.') }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction\n' + message['content'] + '\n\n'}}\n        {%- else %}\n{{'### Response\n' + message['content'] + eos_token + '\n\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{{'### Response\n'}}"
    
    logger.info(f"Tokenizer Class: {tokenizer.__class__.__name__}")
    logger.info(f"PAD Token/Id: {tokenizer.pad_token}/{tokenizer.pad_token_id}")
    logger.info(f"BOS Token/Id: {tokenizer.bos_token}/{tokenizer.bos_token_id}")
    logger.info(f"EOS Token/Id: {tokenizer.eos_token}/{tokenizer.eos_token_id}")
    
    # load data and set data collator
    train_dataset = create_dataset(tokenizer, data_args.data_path, data_args.model_type)
    if data_args.pad_mode == "dynamic_pad":
        data_collator = DataCollatorWithDynamicPad(
            tokenizer, padding=True, max_length=data_args.max_seq_len,
        )
    elif data_args.pad_mode == "dynamic_pack":
        data_collator = DataCollatorWithDynamicPack(
            tokenizer, padding=True, max_length=data_args.max_seq_len,
        )
    else:
        raise ValueError(f"Supported padding modes are `dynamic_pad` and `dynamic_pack`, but you provided {data_args.pad_mode}")

    # get trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # record GPU usage during training
    trainer.add_callback(MemoryUsageCallback(training_args.local_rank, training_args.device))
    
    logger.info("=" * 80)
    if model_args.use_peft_lora:
        logger.info(f"PEFT trainable parameters: {trainer.get_num_trainable_parameters():,}")
    else:
        logger.info(f"FULL trainable parameters: {trainer.get_num_trainable_parameters():,}")
    logger.info("=" * 80)

    # train model
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
