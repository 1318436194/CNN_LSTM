import numpy as np
import torch
from typing import Dict, Any, Union, Optional


def compute_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    metrics: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        predictions: Model predictions (numpy array or tensor)
        targets: Ground truth values (numpy array or tensor)
        metrics: List of metrics to compute (if None, computes all available metrics)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Default metrics to compute
    if metrics is None:
        metrics = ['mse', 'mae', 'rmse', 'mape', 'r2']
    
    # Initialize results dictionary
    results = {}
    
    # Compute metrics
    for metric in metrics:
        if metric.lower() == 'mse':
            # Mean Squared Error
            results['mse'] = float(np.mean((predictions - targets) ** 2))
        
        elif metric.lower() == 'mae':
            # Mean Absolute Error
            results['mae'] = float(np.mean(np.abs(predictions - targets)))
        
        elif metric.lower() == 'rmse':
            # Root Mean Squared Error
            results['rmse'] = float(np.sqrt(np.mean((predictions - targets) ** 2)))
        
        elif metric.lower() == 'mape':
            # Mean Absolute Percentage Error
            epsilon = 1e-10  # Small value to avoid division by zero
            results['mape'] = float(np.mean(np.abs((targets - predictions) / (np.abs(targets) + epsilon))) * 100)
        
        elif metric.lower() == 'r2':
            # R^2 Score (Coefficient of Determination)
            ss_total = np.sum((targets - np.mean(targets)) ** 2)
            ss_residual = np.sum((targets - predictions) ** 2)
            results['r2'] = float(1 - (ss_residual / (ss_total + 1e-10)))
    
    return results


def compute_multistep_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    metrics: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute evaluation metrics for multi-step predictions.
    
    Args:
        predictions: Model predictions with shape (batch_size, forecast_horizon, output_dim)
        targets: Ground truth values with shape (batch_size, forecast_horizon, output_dim)
        metrics: List of metrics to compute (if None, computes all available metrics)
        
    Returns:
        Dictionary containing evaluation metrics for each step
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Get forecast horizon (number of time steps)
    forecast_horizon = predictions.shape[1]
    
    # Initialize results dictionary
    results = {
        'avg': {},  # Average metrics across all steps
        'steps': {}  # Metrics for each step
    }
    
    # Compute metrics for each step
    for step in range(forecast_horizon):
        step_pred = predictions[:, step, :]
        step_target = targets[:, step, :]
        
        # Compute metrics for this step
        step_metrics = compute_metrics(step_pred, step_target, metrics)
        results['steps'][f'step_{step+1}'] = step_metrics
    
    # Compute average metrics across all steps
    for metric_name in results['steps']['step_1'].keys():
        values = [results['steps'][f'step_{i+1}'][metric_name] for i in range(forecast_horizon)]
        results['avg'][metric_name] = float(np.mean(values))
    
    return results 