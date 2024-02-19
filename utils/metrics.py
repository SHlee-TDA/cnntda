"""
This script defines a MetricLogger class along with functions for computing
common metrics used in machine learning tasks, specifically focusing on 
classification and segmentation. 
The MetricLogger class is designed to dynamically log, update, and retrieve 
performance metrics during model training and evaluation, offering 
flexibility to handle both predefined and custom metrics.

The class supports operations such as adding new metrics on-the-fly,
computing metrics based on model outputs and targets, updating metric
values across batches, and resetting metrics for new evaluation phases.
It also facilitates the calculation of standard metrics like Accuracy,
Precision, Recall, F1-Score for classification tasks, and IoU 
(Intersection over Union), Dice Coefficient for segmentation tasks, with
the capability to extend to other metrics as needed.

Predefined metric calculation functions are implemented to work with
PyTorch tensors.
These functions demonstrate how to compute each metric from model
predictions and true labels, ensuring compatibility with the MetricLogger's
dynamic metric handling system.

Key Features:
- Dynamic metric logging for machine learning models, particularly useful
  for tracking performance during training and evaluation.
- Supports classification and segmentation tasks with a set of default
  metrics for each, while also allowing for custom metric functions.
- Utilizes PyTorch for tensor operations, enabling efficient computation
  of metrics directly from model outputs.
- Includes error handling and type checking to ensure robustness and
  reliability of metric computations and logging.

Usage:
The script is intended to be used as part of a larger machine learning
training and evaluation pipeline, where MetricLogger can be instantiated
and used to track model performance over time, across different datasets,
or under various experimental conditions.

Example:
    logger = MetricLogger(task="classification", default_metrics=True)

    # During model evaluation
    outputs = model(x_train)
    targets = y_train
    computed_metrics = logger.compute_metrics(outputs, targets)
    logger.update_metrics(computed_metrics)

    # Retrieve and print metrics
    print(logger.get_metrics(decimal_places=3))

Dependencies:
- torch: Used for handling model outputs and targets as tensors.
- numpy: Used for additional numerical operations.
- sklearn: Provides functions for precision and recall calculations.


Author: Seong-Heon Lee (POSTECH MINDS)
E-mail: shlee0125@postech.ac.kr
Date: 2024-02-20
License: MIT
"""



from typing import Callable, Dict

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score


class MetricLogger:
    """
    A utility class for logging and tracking metrics in machine learning tasks.
    
    This class supports adding, computing, and updating metrics dynamically for
    tasks like classification and segmentation. It is designed to be flexible and
    easy to extend with custom metrics.
    
    Attributes:
        task (str): The type of task (e.g., 'classification', 'segmentation').
        _use_default_metrics (bool): Indicates whether to automatically set up
                                     default metrics for the task.
        metrics (Dict[str, float]): A dictionary to track the accumulated metric values.
        metric_functions (Dict[str, Callable]): A dictionary mapping metric names to
                                                their corresponding computation functions.
    
    Args:
        task (str): The machine learning task for which metrics are to be logged.
        default_metrics (bool, optional): Whether to automatically add a set of
                                          default metrics suitable for the task.
                                          Defaults to True.
    """
    def __init__(self, task: str, default_metrics: bool = True):
        # Validate task type and initialize metrics and metric_functions dictionaries.
        if task not in ["classification", "segmentation"]:
            raise ValueError(f"Unsupported task '{task}'. 
                             Supported tasks are 'classification' and 'segmentation'.")
        self.task = task
        self._use_default_metrics = default_metrics  # Track whether default metrics are used
        self.metrics: Dict[str, float] = {}
        self.metric_functions: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {}  # saving additional metric functions.
        if self._use_default_metrics:
            self.setup_default_metrics()

    def setup_default_metrics(self):
        """
        Automatically adds a set of standard metrics based on the task.
        
        This method initializes the logger with a default set of metrics
        for classification or segmentation tasks. It is called during the
        initialization if `default_metrics` is True.
        """
        if self.task == "classification":
            self.add_metric("Accuracy", compute_accuracy)
            self.add_metric("Recall", compute_recall)
            self.add_metric("Precision", compute_precision)
            self.add_metric("F1-Score", compute_f1)
        elif self.task == "segmentation":
            self.add_metric("IoU", compute_iou)
            self.add_metric("Dice", compute_dice)
        else:
            raise ValueError("Unsupported task specified.")
            
    def add_metric(self, name: str, function: Callable[[torch.Tensor, torch.Tensor], float]):
        """
        Adds a new metric and its computation function to the logger.
        
        Args:
            name (str): The name of the metric to add.
            function (Callable): The function to compute the metric. It should
                                 accept two torch.Tensor arguments: outputs and
                                 targets, and return a float metric value.
        """
        if not callable(function):
            raise TypeError(f"The function for metric '{name}' is not callable.")
        self.metrics[name] = 0.0
        self.metric_functions[name] = function

    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Computes all registered metrics using the provided outputs and targets.
        
        Args:
            outputs (torch.Tensor): The model's outputs for a batch of inputs.
            targets (torch.Tensor): The true labels/targets for a batch of inputs.
        
        Returns:
            Dict[str, float]: A dictionary containing the computed metric values
                              for this batch.
        """
        metric_results: Dict[str, float] = {}
        for name, func in self.metric_functions.items():
            metric_results[name] = func(outputs, targets)
        return metric_results

    def update_metrics(self, computed_metrics: Dict[str, float]):
        """
        Updates the tracked metrics with the results from a new batch.
        
        Args:
            computed_metrics (Dict[str, float]): A dictionary of metrics computed
                                                 for the current batch.
        """
        for name, value in computed_metrics.items():
            if name in self.metrics:
                self.metrics[name] += value
            else:
                raise ValueError(f"Metric '{name}' is not recognized. It must be added before updating.")

    def reset_metrics(self):
        """
        Resets all metrics to their initial state.
        
        This method is useful for starting a new epoch or evaluation phase.
        """
        self.metrics = {name: 0.0 for name in self.metric_functions}
        if self._use_default_metrics:
            self.setup_default_metrics()

    def get_metrics(self, decimal_places: int = 3) -> Dict[str, float]:
        """Retrieve the current metrics, formatted to a specified number of 
        decimal places.
        
        Args:
            decimal_places (int, optional): The number of decimal places to 
                                            format the metric values.
                                            Defaults to 3.
        Returns:
            dict: A dictionary of the metrics, with values formatted to the 
                  specified number of decimal places.
        """
        return {key: round(value, decimal_places) for key, value in self.metrics.items()}


# Metrics for Classification tasks
def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the accuracy of classification predictions.
    
    Args:
        outputs (torch.Tensor): The logits from the model,
                                where the shape is of [batch_size, num_classes].
        targets (torch.Tensor): The ground truth labels for each input sample,
                                where the shape is of [batch_size].
    
    Returns:
        float: The accuracy of predictions, defined as the proportion of
               correct predictions over the total number of predictions.
    """
    if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Outputs and targets must be torch.Tensor objects.")
    if outputs.size(0) != targets.size(0):
        raise ValueError("Outputs and targets must have the same batch size.")
    
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return correct / total


def compute_precision(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the precision of classification predictions.
    
    Precision is defined as the number of true positives over the number of
    true positives plus the number of false positives.
    This function computes the macro-average precision, which calculates
    precision independently for each class and then takes the average.
    
    Args:
        outputs (torch.Tensor): The logits from the model,
                                where the shape is of [batch_size, num_classes].
        targets (torch.Tensor): The ground truth labels for each input sample,
                                where the shape is of [batch_size].

    
    Returns:
        float: The macro-average precision of the predictions.
    """
    if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Outputs and targets must be torch.Tensor objects.")
    if outputs.size(0) != targets.size(0):
        raise ValueError("Outputs and targets must have the same batch size.")
    
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()
    precision = precision_score(targets, predicted,
                                average='macro',
                                zero_division=1)
    return precision


def compute_recall(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the recall of classification predictions.
    
    Recall is defined as the number of true positives over the number of
    true positives plus the number of false negatives. 
    This function computes the macro-average recall, which calculates recall 
    independently for each class and then takes the average.
    
    Args:
        outputs (torch.Tensor): The logits from the model,
                                where the shape is of [batch_size, num_classes].
        targets (torch.Tensor): The ground truth labels for each input sample,
                                where the shape is of [batch_size].
    
    Returns:
        float: The macro-average recall of the predictions.
    """
    if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Outputs and targets must be torch.Tensor objects.")
    if outputs.size(0) != targets.size(0):
        raise ValueError("Outputs and targets must have the same batch size.")
    
    _, predicted = torch.max(outputs.data, 1)
    
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()
    
    recall = recall_score(targets, predicted, 
                          average='macro', 
                          zero_division=1)
    return recall


def compute_f1(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the F1 score from precision and recall.
    
    The F1 score is the harmonic mean of precision and recall, providing a 
    balance between the two metrics. 
    It is a useful measure when the class distribution is imbalanced.
    
    Args:
        outputs (torch.Tensor): The logits from the model,
                                where the shape is of [batch_size, num_classes].
        targets (torch.Tensor): The ground truth labels for each input sample,
                                where the shape is of [batch_size].
    Returns:
        float: The F1 score, which ranges from 0 to 1.
    """
    if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Outputs and targets must be torch.Tensor objects.")
    if outputs.size(0) != targets.size(0):
        raise ValueError("Outputs and targets must have the same batch size.")
    
    precision = compute_precision(outputs, targets)
    recall = compute_recall(outputs, targets)
    if precision + recall == 0:  # Avoid division by zero
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# Metrics for Segmentation tasks
def compute_iou(outputs: torch.Tensor, targets: torch.Tensor, threshold=0.5) -> float:
    """
    Compute the Intersection over Union (IoU) for binary segmentation tasks.
    IoU is a measure of the overlap between two binary masks: the predicted 
    mask and the ground truth mask.
    
    Args:
        outputs (torch.Tensor): The predicted segmentation masks, 
                                where the shape is of [batch_size, height, width].
        targets (torch.Tensor): The ground truth segmentation masks, 
                                where the shape is of [batch_size, height, width].
        threshold (float, optional): The threshold to convert probabilities in
                                     the predicted masks to binary values (0 or 1).
                                     Defaults to 0.5.
    
    Returns:
        float: The average IoU score across all the samples in the batch.
    """
    batch_size = outputs.shape[0]
    iou_scores = []

    for i in range(batch_size):
        output = outputs[i] > threshold
        target = targets[i]

        intersection = torch.logical_and(output, target)
        union = torch.logical_or(output, target)
        iou = torch.sum(intersection) / torch.sum(union)
        iou_scores.append(iou.item())

    return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0


def compute_dice(outputs: torch.Tensor, targets: torch.Tensor, threshold=0.5) -> float:
    """
    Compute the Dice Coefficient (also known as Sørensen-Dice index)
    for binary segmentation tasks. The Dice Coefficient is a measure of 
    the overlap between two samples. It is defined as twice the area of 
    intersection divided by the sum of areas of both samples.
    
    Args:
        outputs (torch.Tensor): The predicted segmentation masks,
                                where the shape is of [batch_size, height, width].
        targets (torch.Tensor): The ground truth segmentation masks,
                                where the shape is of [batch_size, height, width].
        threshold (float, optional): The threshold to convert probabilities in
                                     the predicted masks to binary values (0 or 1).
                                     Defaults to 0.5.
    
    Returns:
        float: The average Dice Coefficient across all the samples in the batch.
    """
    batch_size = outputs.shape[0]
    dice_scores = []

    for i in range(batch_size):
        output = outputs[i] > threshold
        target = targets[i]

        intersection = torch.logical_and(target, output)
        dice = 2 * torch.sum(intersection) / (torch.sum(target) + torch.sum(output))
        dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0