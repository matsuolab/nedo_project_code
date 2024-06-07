import torch
import wandb
import os
import time
import math
from typing import Any

def log_model_info(model: torch.nn.Module) -> None:
    model_config: dict[str, Any] = {}
    model_config["activation_function"] = "silu"
    model_config["hidden_size"] = model.llma.args.dim
    model_config["num_hidden_layers"] = model.llma.args.n_layers
    model_config["intermediate_size"] =model.llma.args.hidden_dim
    model_config["max_position_embeddings"] = model.llma.args.max_seq_len 
    model_config["num_attention_heads"] = model.llma.args.n_heads
    model_config["num_key_value_heads"] = model.llma.args.n_kv_heads
    model_config["vocab_size"] = model.tokenizer.n_words
    model_config["model_peft"] = model.is_peft
    model_config["rope_theta"] = model.llma.args.rope_theta
    model_config["norm_eps"] = model.llma.args.norm_eps
    model_config["load_balancing_weight"] = model.llma.args.load_balancing_weight
    
    print(f"model config: {model.llma.args}")
    
    wandb.config.update(model_config)

    # distributed training info
    world_size = int(os.environ["WORLD_SIZE"])
    wandb.config.update({"world_size": world_size})


def log_wandb(
    #real_batch_size: int,
    #real_seq_len: int,
    model: torch.nn.Module,
    accumulation_loss: float,
    load_balancing_loss: float,
    iteration: int,
    lr: float,
    #gradient_accumulation_steps: int,
    #world_size: int,
    #iteration_start_time: float,
) -> None:
    wandb_stats: dict[str, Any] = {}

    # training info
    wandb_stats["training/loss"] = accumulation_loss
    wandb_stats["training/load_balancing_loss"] = load_balancing_loss
    wandb_stats["training/perplexity"] = math.exp(accumulation_loss)
    wandb_stats["training/lr"] = lr
    # utils info
    wandb_stats["utils/iteration"] = iteration
    
    
    """
    batch_size: int = real_batch_size
    sequence_length: int = real_seq_len
    wandb_stats["utils/batch_size"] = batch_size
    wandb_stats["utils/global_batch_size"] = batch_size * world_size * gradient_accumulation_steps
    wandb_stats["utils/seq_len"] = sequence_length
    wandb_stats["utils/gradient_accumulation_steps"] = gradient_accumulation_steps
     
    #stats
    iteration_elapsed_time = time.perf_counter() - iteration_start_time

    tokens_per_sec = batch_size * sequence_length * gradient_accumulation_steps / iteration_elapsed_time * world_size
    wandb_stats["stats/1_iteration_time"] = iteration_elapsed_time
    wandb_stats["stats/tokens_per_sec"] = tokens_per_sec
    wandb_stats["stats/tokens_per_sec_per_gpu"] = tokens_per_sec / world_size

    checkpoint_activations_factor = 3

    num_layers: int = model.llma.args.n_layers
    hidden_size: int = model.llma.args.dim
    vocab_size: int = model.tokenizer.n_words
    intermediate_size: int = model.llma.args.hidden_dim

    #activation_function_factor: int = 4  # GELU
    activation_function_factor = 4 + 2  # SWiGLU (upscaling + down scaling)

    batch_size = batch_size * gradient_accumulation_steps
    num_query_groups: int = model.llma.args.n_heads / model.llma.args.n_kv_heads

    # tflops calculation
    flops_per_iteration: float = checkpoint_activations_factor * ((
        (2 + (2 * 3) + activation_function_factor * (intermediate_size / hidden_size)) * batch_size * sequence_length * num_layers * (hidden_size**2)
    ) + (
        ((  # Attention matrix & attention over values
            4 * batch_size * (sequence_length ** 2) * hidden_size
        ) / num_query_groups
        ) +  # noqa: W504
        # lm-head: logit layer
        2 * batch_size * sequence_length * hidden_size * vocab_size)
    )
    tflops: float = flops_per_iteration / (iteration_elapsed_time * (10**12))
    wandb_stats["stats/tflops"] = tflops
    """
    
    wandb.log(wandb_stats, step=iteration)

    #print("------------------------------------------------------------------")
    #print(f"iteration: {iteration} , TFLOPS: {tflops}, Tokens per sec: {tokens_per_sec}, Loss: {accumulation_loss}, load balancing loss: {load_balancing_loss}")
    
