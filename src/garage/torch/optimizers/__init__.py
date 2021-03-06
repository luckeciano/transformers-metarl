"""PyTorch optimizers."""
from garage.torch.optimizers.conjugate_gradient_optimizer import (
    ConjugateGradientOptimizer)
from garage.torch.optimizers.differentiable_sgd import DifferentiableSGD
from garage.torch.optimizers.optimizer_wrapper import OptimizerWrapper, WarmupOptimizerWrapper, LRDecayOptimizerWrapper

__all__ = [
    'OptimizerWrapper', 'ConjugateGradientOptimizer', 'DifferentiableSGD', 'WarmupOptimizerWrapper', 'LRDecayOptimizerWrapper'
]
