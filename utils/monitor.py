import gc
import torch
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


# Converting Bytes to Gigabytes
def b2gb(x):
    return x / 2**30


class MemoryUsageCallback(TrainerCallback):
    def __init__(self, local_rank, device):
        self.local_rank = local_rank
        self.device = device
        self.max_memory = 0
        self.total_memory = 0
        self.count = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        gc.collect()
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats() # reset the peak gauge to zero
        self.gpu_begin = torch.cuda.memory_allocated()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.count += 1
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        maximun_memory = torch.cuda.max_memory_allocated()
        self.max_memory = max(self.max_memory, maximun_memory)
        self.total_memory += maximun_memory

        torch.cuda.reset_peak_memory_stats() # reset the peak gauge to zero

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.count > 0:
            self.average_memory = self.total_memory / self.count
        else:
            self.average_memory = 0
        print(f"[Process rank: {self.local_rank}, device: {self.device}] Memory before entering the train: {b2gb(self.gpu_begin):.2f} GB")
        print(f"[Process rank: {self.local_rank}, device: {self.device}] Average Memory during the train: {b2gb(self.average_memory):.2f} GB")
        print(f"[Process rank: {self.local_rank}, device: {self.device}] Total Peak Memory consumed during the train: {b2gb(self.max_memory):.2f} GB")

