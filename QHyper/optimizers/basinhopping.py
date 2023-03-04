from scipy.optimize import basinhopping
import numpy as np

from typing import Callable, Any

from .optimizer import HyperparametersOptimizer, ArgsType, Optimizer, Wrapper


class Basinhopping(HyperparametersOptimizer):
    """Implementation of Cross Entropy Method for hyperparamter tuning
    """
    def __init__(self, niter: int, maxfun: int, bounds: list[tuple[float, float]] = None) -> None:
        self.niter = niter
        self.maxfun = maxfun
        self.bounds = np.array(bounds)

    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer,
        init: ArgsType, 
        hyperparams_init: ArgsType, 
        evaluation_func: Callable[[ArgsType], Callable[[ArgsType], float]] = None,
        bounds: list[float] = None,
        **kwargs: Any
    ) -> ArgsType:
        """Returns hyperparameters which leads to the lowest values returned by optimizer 1
        
        Parameters
        ----------
        func_creator : Callable[[ArgsType], Callable[[ArgsType], float]]
            function, which receives hyperparameters, and returns  
            function which will be optimized using optimizer
        optimizer : Optimizer
            object of class Optimizer
        init : ArgsType
            initial args for optimizer
        hyperparams_init : ArgsType
            initial hyperparameters
        bounds : list[float]
            bounds for hyperparameters (default None)
        evaluation_func : Callable[[ArgsType], Callable[[ArgsType], float]]
            function, which receives hyperparameters, and returns 
            function which receives params and return evaluation
        kwargs : Any
            allow additional arguments passed to scipy.optimize.Shgo
        Returns
        -------
        ArgsType
            hyperparameters which leads to the lowest values returned by optimizer
        """

        # wrapper = Wrapper(func_creator, optimizer, evaluation_func, init)
        # def wrapper(params):
        #     weights = params[:len(hyperparams_init)]
        #     angles = np.array(params[len(hyperparams_init):]).reshape(init.shape)

        #     return evaluation_func(weights, angles)

        # init_params = list(hyperparams_init) + list(np.array(init).flatten())

        # result = basinhopping(
        #     wrapper, init_params, niter=self.niter, 
        #     minimizer_kwargs={
        #         'options': {'maxiter': self.maxfun},
        #         'bounds': self.bounds
        #     }, **kwargs)

        # return result.fun, np.array(result.x[len(hyperparams_init):]).reshape(init.shape), result.x[:len(hyperparams_init)]
        def wrapper(angles):
            return evaluation_func(hyperparams_init, angles.reshape(init.shape))

        init_params = np.array(init).flatten()

        result = basinhopping(
            wrapper, init_params, niter=self.niter, 
            minimizer_kwargs={
                'options': {'maxfun': self.maxfun},
                'bounds': self.bounds[2:]
            }, **kwargs)

        return result.fun, np.array(result.x).reshape(init.shape), hyperparams_init

