import dataclasses

from abc import abstractmethod
import numpy as np

import numpy.typing as npt
from typing import Callable


@dataclasses.dataclass
class OptimizationResult:
    """
    Dataclass for storing the results of an optimization run.
    
    Attributes
    ----------
    value : float
        The minimum function value found by the optimization algorithm.
    params : numpy.ndarray
        The optimal point found by the optimization algorithm.
    """

    value: float
    params: npt.NDArray[np.float64]


class Optimizer:
    """
    Abstract base class for optimizers.
    """

    @abstractmethod
    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], OptimizationResult],
        init: npt.NDArray[np.float64]
    ) -> OptimizationResult:
        """
        Abstract method that minimizes the given function using the implemented optimization algorithm.

        Parameters
        ----------
        func : callable
            The objective function to be minimized.
        init : numpy.ndarray
            The initial point for the optimization algorithm.

        Returns
        -------
        tuple
            A tuple containing the minimum function value and the corresponding optimal point.

        Raises
        ------
        NotImplementedError
            If the `minimize` method is not implemented by the derived class.
        """
        ...
