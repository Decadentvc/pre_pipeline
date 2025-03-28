from pipeline.PrototypeSingleton import PrototypeSingleton
from .policy import Policy
from algorithm import space as ALGORITHM_SPACE
from .objective import objective_pipeline, objective_algo
from hyperopt import tpe, fmin, Trials

import functools
import numpy as np


class Split(Policy):
    """Policy of optimization that performs the optimization subsequentially, specifically: data pre-processing and ML algorithm.
    """

    def run(self, X, y):
        super(Split, self).run(X, y)
        current_pipeline_configuration = {}
        current_algo_configuration = {}
        trials_pipelines = Trials()
        trials_algo = Trials()

        if self.config["experiment"] == "pipeline_impact":
            if self.config['time'] - self.config['step_pipeline'] > 0:
                current_algo_configuration = self._optimize_algorithm(X, y, current_pipeline_configuration, trials_algo)
            if self.config['step_pipeline'] > 0:
                self._optimize_pipeline(X, y, current_algo_configuration, trials_pipelines)
        else:
            if self.config['step_pipeline'] > 0:
                current_pipeline_configuration = self._optimize_pipeline(X, y, current_algo_configuration, trials_pipelines)
            if self.config['time'] - self.config['step_pipeline'] > 0:
                self._optimize_algorithm(X, y, current_pipeline_configuration, trials_algo)

    def _get_budget(self, phase):
        """Gets the budget according to the phase at hand

        Args:
            phase: either pipeline or algorithm

        Returns:
            int: max evaluations
            int: max time budget
        """
        if phase == 'algorithm':
            budget = self.config['time'] - self.config['step_pipeline']
        else:
            budget = self.config['step_pipeline']

        if self.config["experiment"] == "pipeline_impact":
            max_evals = budget
            max_time = budget if self.config['toy'] else 80000
        else:
            max_evals = budget if self.config['toy'] else None
            max_time = budget
        
        # print(f"toy: {self.config['toy']}, max_evals: {max_evals}, max_time: {max_time}")
        return max_evals, max_time
            

    def _optimize_algorithm(self, X, y, current_pipeline_configuration, trials_algo):
        """Performs the optimization on the ML algorithm at hand.

        Args:
            X: data items.
            y: labels.
            current_pipeline_configuration: best pipeline configuration.
            trials_algo: data structure for data saving.

        Returns:
            dict: best algorithm configuration.
        """
        print('## Algorithm')
        obj_algo = functools.partial(objective_algo,
                current_pipeline_config=current_pipeline_configuration,
                algorithm=self.config['algorithm'],
                X=X,
                y=y,
                context=self.context,
                config=self.config)
        max_evals, max_time = self._get_budget('algorithm')
        fmin(fn=obj_algo,
            space=ALGORITHM_SPACE.get_domain_space(self.config['algorithm']),
            algo=tpe.suggest,
            max_evals=max_evals,
            max_time=max_time,
            trials=trials_algo,
            show_progressbar=False,
            verbose=0,
            rstate=np.random.RandomState(self.config['seed']) 
        )

        best_config = self.context['best_config']
        current_algo_configuration = best_config['algorithm']
        super(Split, self).display_step_results(best_config)
        return current_algo_configuration

    def _optimize_pipeline(self, X, y, current_algo_configuration, trials_pipelines):
        """Performs the optimization on the prototype at hand.

        Args:
            X: data items.
            y: labels.
            current_algo_configuration: best algorithm configuration.
            trials_algo: data structure for data saving.

        Returns:
            dict: best pipeline configuration.
        """
        print('## Data Pipeline')
        obj_pl = functools.partial(objective_pipeline,
                current_algo_config=current_algo_configuration,
                algorithm=self.config['algorithm'],
                X=X,
                y=y,
                context=self.context,
                config=self.config)
        max_evals, max_time = self._get_budget('pipeline')
        fmin(
            fn=obj_pl, 
            space=self.PIPELINE_SPACE,
            algo=tpe.suggest, 
            max_evals=max_evals,
            max_time=max_time,
            trials=trials_pipelines,
            show_progressbar=False,
            verbose=0,
            rstate=np.random.RandomState(self.config['seed']) 
        )

        best_config = self.context['best_config']
        current_pipeline_configuration = best_config['pipeline']
        super(Split, self).display_step_results(best_config)
        return current_pipeline_configuration
