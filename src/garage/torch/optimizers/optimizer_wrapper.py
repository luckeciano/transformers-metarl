"""A PyTorch optimizer wrapper that compute loss and optimize module."""
from garage import make_optimizer
from garage.np.optimizers import BatchDataset
import torch
import pytorch_warmup as warmup
import numpy as np


class OptimizerWrapper:
    """A wrapper class to handle torch.optim.optimizer.

    Args:
        optimizer (Union[type, tuple[type, dict]]): Type of optimizer
            for policy. This can be an optimizer type such as
            `torch.optim.Adam` or a tuple of type and dictionary, where
            dictionary contains arguments to initialize the optimizer.
            e.g. `(torch.optim.Adam, {'lr' : 1e-3})`
            Sample strategy to be used when sampling a new task.
        module (torch.nn.Module): Module to be optimized.
        max_optimization_epochs (int): Maximum number of epochs for update.
        minibatch_size (int): Batch size for optimization.

    """

    def __init__(self,
                 optimizer,
                 module,
                 max_optimization_epochs=1,
                 minibatch_size=None):
        self._optimizer = make_optimizer(optimizer, module=module)
        self._max_optimization_epochs = max_optimization_epochs
        self._minibatch_size = minibatch_size

    def get_minibatch(self, *inputs):
        r"""Yields a batch of inputs.

        Notes: P is the size of minibatch (self._minibatch_size)

        Args:
            *inputs (list[torch.Tensor]): A list of inputs. Each input has
                shape :math:`(N \dot [T], *)`.

        Yields:
            list[torch.Tensor]: A list batch of inputs. Each batch has shape
                :math:`(P, *)`.

        """
        batch_dataset = BatchDataset(inputs, self._minibatch_size)

        for _ in range(self._max_optimization_epochs):
            for dataset in batch_dataset.iterate():
                yield dataset

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        self._optimizer.zero_grad()

    def step(self, **closure):
        """Performs a single optimization step.

        Arguments:
            **closure (callable, optional): A closure that reevaluates the
                model and returns the loss.

        """
        self._optimizer.step(**closure)

class WarmupOptimizerWrapper(OptimizerWrapper):
    def __init__(self,
                 optimizer,
                 module,
                 n_epochs,
                 steps_per_epoch,
                 max_optimization_epochs=1,
                 min_lr_factor = 0.1, #final lr will be nominal_lr * min_lr_factor
                 minibatch_size=None):
        super().__init__(optimizer, module, max_optimization_epochs, minibatch_size)

        n_steps = n_epochs * steps_per_epoch
        milestones = np.arange(2 * n_steps // 3, n_steps, steps_per_epoch)
        gamma = (min_lr_factor) ** (1 / len(milestones))
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=milestones, gamma=gamma)
        self._warmup_scheduler =  warmup.LinearWarmup(self._optimizer, n_steps//3)
        self._warmup_scheduler.last_step = -1 # initialize the step counter
        self._epoch = 0

    def zero_grad(self):
        self._lr_scheduler.step(self._epoch)
        self._warmup_scheduler.dampen()
        self._optimizer.zero_grad()
        self._epoch += 1
