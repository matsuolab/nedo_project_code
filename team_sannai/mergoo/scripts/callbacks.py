import os
import time
import wandb
from transformers import TrainerCallback


class ComputeThroughputCallback(TrainerCallback):
    def __init__(self, vocab_size = 32000, seq_length = 4096, num_layers = 32, hidden_size = 4096, world_size = 1, log_steps = 100, use_activation_checkpointing = False):
        super().__init__()
        self.start_time = None
        self.total_time = 0
        self.iterations = 0

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.world_size = world_size

        self.log_steps = log_steps
        self.use_activation_checkpointing = use_activation_checkpointing

    def on_step_begin(self, args, state, control, **kwargs):
        """イテレーションが開始する前に現在時刻を記録します。"""
        if state.global_step % args.gradient_accumulation_steps == 0:
            self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        """イテレーションが終了した後に経過時間を計算し、TFLOPsを計算するために使用します。"""
        if self.start_time is not None and (state.global_step % args.gradient_accumulation_steps == 0 or state.global_step == state.max_steps):
            time_per_iter = time.time() - self.start_time
            self.total_time += time_per_iter
            self.iterations += 1

            # average_time_per_iter = self.total_time / self.iterations
            elapsed_time_per_iter = self.total_time / self.iterations
            print(f"Elapsed Time per Iteration: {elapsed_time_per_iter:.2f} seconds")
            
            effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * self.world_size
            tflops = self.compute_tflops(effective_batch_size, elapsed_time_per_iter)
            
            # samples_per_sec = self.batch_size / elapsed_time_per_iter
            samples_per_sec = effective_batch_size / elapsed_time_per_iter
            
            tokens_per_sec = samples_per_sec * self.seq_length
            metrics = {
                    "tflops": tflops,
                    "samples_per_sec": samples_per_sec,
                    "tokens_per_sec": tokens_per_sec
            }
            consumed_tokens = effective_batch_size * self.seq_length
            if int(os.environ['LOCAL_RANK']) == 0:
                print(f"tflops: {tflops:.2f}, samples_per_sec: {samples_per_sec}, tokens_per_sec: {tokens_per_sec}, consumed_tokens: {consumed_tokens}")

            if wandb is not None and getattr(wandb, 'run', None) is not None:
                wandb.log(metrics)

            # 次の計測のために時刻をリセット
            self.start_time = None

    def compute_tflops(self, batch_size, elapsed_time_per_iter):
        """
        TFLOPsを計算する関数。
        
        Args:
            args (TrainingArguments): トレーニング設定が含まれるオブジェクト。
            batch_size (int): バッチサイズ。
            elapsed_time_per_iter (float): イテレーション毎の経過時間（秒）。
        
        Returns:
            float: TFLOPs値。
        """
        # チェックポイント活用の影響を計算
        checkpoint_activations_factor = 3
        if self.use_activation_checkpointing:
            checkpoint_activations_factor = 4

        # イテレーション毎のFLOPSを計算
        flops_per_iteration = (
            (24 * checkpoint_activations_factor * batch_size * self.seq_length * self.num_layers * (self.hidden_size ** 2)) *
            (1. + (self.seq_length / (6. * self.hidden_size)) + (self.vocab_size / (16. * self.num_layers * self.hidden_size)))
        )

        # TFLOPsを計算
        tflops = flops_per_iteration / (elapsed_time_per_iter * self.world_size * (10 ** 12))

        return tflops
    

class TokenCountCallback(TrainerCallback):
    def __init__(self, max_token_count):
        self.max_token_count = max_token_count
        self.token_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.token_count += args.per_device_train_batch_size * args.gradient_accumulation_steps
        if self.token_count >= self.max_token_count:
            print("current tokens: ", self.token_count)
            print(f"指定されたトークン数 {self.max_token_count} に到達。学習を終了します。")
            control.should_training_stop = True
