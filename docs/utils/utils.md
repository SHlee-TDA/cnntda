# MetricLogger Documentation

## Overview

`MetricLogger` is a utility class designed for logging and tracking metrics in machine learning tasks, supporting both classification and segmentation tasks. It facilitates the dynamic addition, computation, and updating of metrics, offering flexibility to handle predefined and custom metrics.

## Usage

To use `MetricLogger` in your project, import the class and initialize it with your task type:

```python
from utils.metrics import MetricLogger

# Initialize for a classification task
logger = MetricLogger(task="classification", default_metrics=True)
```

### Adding Custom Metrics

You can add custom metrics by providing a metric name and a computation function:

```python
logger.add_metric("CustomMetric", custom_metric_function)
```

### Computing Metrics

To compute metrics for a batch of outputs and targets:

```python
outputs = model(inputs)
targets = ground_truth_labels
computed_metrics = logger.compute_metrics(outputs, targets)
```

### Updating and Retrieving Metrics

Update the logged metrics with the computed values and retrieve them:

```python
logger.update_metrics(computed_metrics)
metrics = logger.get_metrics(decimal_places=2)
```

## API Reference

### `MetricLogger` Class
- `__init__(self, task: str, default_metrics: bool = True)`: Initializes the MetricLogger.
- `add_metric(self, name: str, function: Callable)`: Adds a new metric.
- `compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor)`: Computes all registered metrics.
- `update_metrics(self, computed_metrics: Dict[str, float])`: Updates the tracked metrics.
- `reset_metrics(self)`: Resets all metrics to their initial state.
- `get_metrics(self, decimal_places: int = 3)`: Retrieves the current metrics.

### Metric Calculation Functions
- `compute_accuracy(outputs, targets)`: Computes accuracy for classification tasks.
- `compute_precision(outputs, targets)`: Computes precision for classification tasks.
- `compute_recall(outputs, targets)`: Computes recall for classification tasks.
- `compute_f1(outputs, targets)`: Computes the F1 score from precision and recall.
- `compute_iou(outputs, targets, threshold=0.5)`: Computes IoU for segmentation tasks.
- `compute_dice(outputs, targets, threshold=0.5)`: Computes Dice Coefficient for segmentation tasks.
