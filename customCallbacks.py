import pytorch_lightning as pl
import torch
import statistics

class InferenceTimeCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.test_batch_start_event = torch.cuda.Event(enable_timing=True)
        self.test_batch_end_event = torch.cuda.Event(enable_timing=True)

        self.inference_times = []

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # Record start event
        self.test_batch_start_event.record()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # Record end event and synchronize
        self.test_batch_end_event.record()
        torch.cuda.synchronize()  # Wait for the events to complete
        batch_inference_time = self.test_batch_start_event.elapsed_time(self.test_batch_end_event) / 1000  # Convert ms to seconds
        self.inference_times.append(batch_inference_time)

    def on_test_end(self, trainer, pl_module):
        # Calculate and print average inference time at the end
        print(f"Average inference time per batch: {statistics.mean(self.inference_times):.4f} seconds")
        print(f"Median inference time per batch: {statistics.median(self.inference_times):.4f} seconds")
        print(f"Total inference time: {sum(self.inference_times):.4f} seconds")