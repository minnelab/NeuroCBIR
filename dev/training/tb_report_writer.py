from typing import Union
from torch.utils.tensorboard.writer import SummaryWriter

class AverageLoss:
    """
    Utility class to track losses
    and metrics during training.
    """

    def __init__(self):
        self.losses_accumulator = {}
    
    def put(self, loss_key:str, loss_value:Union[int,float]) -> None:
        """
        Store value

        Args:
            loss_key (str): Metric name
            loss_value (int | float): Metric value to store
        """
        if loss_key not in self.losses_accumulator:
            self.losses_accumulator[loss_key] = []
        self.losses_accumulator[loss_key].append(loss_value)
    
    def pop_avg(self, loss_key:str) -> float:
        """
        Average the stored values of a given metric

        Args:
            loss_key (str): Metric name

        Returns:
            float: average of the stored values
        """
        if loss_key not in self.losses_accumulator:
            return None
        losses = self.losses_accumulator[loss_key]
        self.losses_accumulator[loss_key] = []
        return sum(losses) / len(losses)
    
    def to_tensorboard(self, writer: SummaryWriter, step: int):
        """
        Logs the average value of all the metrics stored 
        into Tensorboard.

        Args:
            writer (SummaryWriter): Tensorboard writer
            step (int): Tensorboard logging global step 
        """
        for metric_key in self.losses_accumulator.keys():
            writer.add_scalar(metric_key, self.pop_avg(metric_key), step)