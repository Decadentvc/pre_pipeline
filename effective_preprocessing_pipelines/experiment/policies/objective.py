import hashlib
import json
import time
from multiprocessing import Process, Pipe
import sys

from hyperopt import STATUS_OK, STATUS_FAIL
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate

from algorithm import space as ALGORITHM_SPACE
from pipeline.prototype import pipeline_conf_to_full_pipeline, get_baseline


def objective(pipeline_config, algo_config, algorithm, X, y, context, config, step):
    """Function that is executed in the optimization process (to maximize).

    Args:
        pipeline_config: prototype to optimize.
        algo_config: algorithm to optimize.
        algorithm: legacy.
        X: data items.
        y: labels.
        context: data structure to save the progresses.
        config: extra information.
        step: either algorithm or pipeline.

    Returns:
        dict: summary of the optimization
    """
    pipeline_hash = hashlib.sha1(json.dumps(pipeline_config, sort_keys=True).encode()).hexdigest()
    algorithm_hash = hashlib.sha1(json.dumps(algo_config, sort_keys=True).encode()).hexdigest()
    item_hash = {
        'pipeline': pipeline_hash,
        'algorithm': algorithm_hash,
        'config': hashlib.sha1(str(pipeline_hash + algorithm_hash).encode()).hexdigest()
    }

    item = {
        'pipeline': pipeline_config,
        'algorithm': algo_config,
        'step': step
    }

    pipeline, operators = pipeline_conf_to_full_pipeline(
        pipeline_config, 
        ALGORITHM_SPACE.algorithms.get(config['algorithm']), 
        config['seed'], 
        algo_config
    )
  
    history_index = context['history_index'].get(item_hash['config'])
    if history_index is not None:
        return context['history'][history_index]

    start = time.time()
    try:
        scores = cross_validate(pipeline, 
                X,
                y,
                scoring = ["balanced_accuracy"],
                cv=10,
                n_jobs=-1,
                return_estimator=False,
                return_train_score=False,
                verbose=0)
        score = np.mean(scores['test_balanced_accuracy']) // 0.0001 / 10000
        std = np.std(scores['test_balanced_accuracy']) // 0.0001 / 10000
        status = STATUS_OK
    except Exception as e:
        score = 0.
        std = 0.
        status = STATUS_FAIL
        print(e)
    stop = time.time()

    iteration_number = len(context['history'])
    item.update({
        'start_time': start,
        'stop_time': stop,
        'duration': stop - start,
        'loss': 1 - score, 
        'status': status, 
        'score': score,
        'score_std': std,
        'iteration': iteration_number,
        'config_hash': item_hash,
        'max_history_score': context['max_history_score'],
        'max_history_score_std': context['max_history_score_std'],
        'max_history_step': context['max_history_step'],
        'step': step
    })

    if context['max_history_score'] < score:
        item['max_history_score'] = score
        context['max_history_score'] = score
        context['max_history_score_std'] = std
        item['max_history_score_std'] = std
        context['max_history_step'] = step
        item['max_history_step'] = step
        context['best_config'] = item


    # Update hash index
    context['history_hash'].append(item_hash['config'])
    context['history_index'][item_hash['config']] = iteration_number
    context['iteration'] = iteration_number

    context['history'].append(item)

    print('Best score: {} ({}) [{}] | Score: {} ({}) [{}]'.format(
        item['max_history_score'],
        item['max_history_score_std'],
        item['max_history_step'][0].upper(),
        item['score'],
        item['score_std'],
        item['step'][0].upper(),
        )
    )
    return item

def objective_pipeline(pipeline_config, current_algo_config, algorithm, X, y, context, config):
    """Performs the optimization on the data pre-processing pipeline.

    Args:
        pipeline_config: prototype to optimize.
        algo_config: algorithm to optimize.
        algorithm: legacy.
        X: data items.
        y: labels.
        context: data structure to save the progresses.
        config: extra information.

    Returns:
        dict: summary of the optimization
    """
    return objective(pipeline_config, current_algo_config, algorithm, X, y, context, config, step='pipeline')

def objective_algo(algo_config, current_pipeline_config, algorithm, X, y, context, config):
    """Performs the optimization on the data pre-processing pipeline.

    Args:
        pipeline_config: prototype to optimize.
        algo_config: algorithm to optimize.
        algorithm: legacy.
        X: data items.
        y: labels.
        context: data structure to save the progresses.
        config: extra information.

    Returns:
        dict: summary of the optimization
    """
    return objective(current_pipeline_config, algo_config, algorithm, X, y, context, config, step='algorithm')

def objective_joint(wconfig, algorithm, X, y, context, config):
    """Performs the optimization of the data pre-processing pipeline and the ML algorithm jointly.

    Args:
        wconfig: joint configuration of pipeline and algorithm.
        algorithm: legacy
        X: data items.
        y: labels.
        context: data structure to save the progresses.
        config: extra information.

    Returns:
        _type_: _description_
    """
    return objective(wconfig['pipeline'], wconfig['algorithm'], algorithm, X, y, context, config, step='joint')


def get_baseline_score(algorithm, X, y, seed):
    """Computes the baseline (no optimization at all).

    Args:
        algorithm: used algorithm.
        X: data items.
        y: labels.
        seed: seed for reproducibility.

    Returns:
        float: mean of the accuracies among the folds.
        float: std of the accuracies among the folds.
    """
    pipeline, _ = pipeline_conf_to_full_pipeline(
        get_baseline(), 
        ALGORITHM_SPACE.algorithms.get(algorithm), 
        seed, 
        {}
    )
    try:
        scores = cross_validate(pipeline, 
                        X,
                        y,
                        # scoring=["precision_macro", "recall_macro", "f1_macro", "roc_auc", "accuracy", "balanced_accuracy"],
                        scoring=["balanced_accuracy"],
                        cv=10,
                        n_jobs=-1,
                        return_estimator=False,
                        return_train_score=False,
                        verbose=0)
        score = np.mean(scores['test_balanced_accuracy']) // 0.0001 / 10000
        std = np.std(scores['test_balanced_accuracy']) // 0.0001 / 10000
    except Exception as e:
        score = 0.
        std = 0.
        print(e)
    return score, std


def wrap_cost(cost_fn, timeout=None, iters=1, verbose=0):
    """Wrap cost function to execute trials safely on a separate process.
    Parameters
    ----------
    cost_fn : callable
        The cost function (aka. objective function) to wrap. It follows the
        same specifications as normal Hyperopt cost functions.
    timeout : int
        Time to wait for process to complete, in seconds. If this time is
        reached, the process is re-tried if there are remaining iterations,
        otherwise marked as a failure. If ``None``, wait indefinitely.
    iters : int
        Number of times to allow the trial to timeout before marking it as
        a failure due to timeout.
    verbose : int
        How verbose this function should be. 0 is not verbose, 1 is verbose.
    Example
    -------
    def objective(args):
        case, val = args
        return val**2 if case else val
    space = [hp.choice('case', [False, True]), hp.uniform('val', -1, 1)]
    safe_objective = wrap_cost(objective, timeout=2, iters=2, verbose=1)
    best = hyperopt.fmin(safe_objective, space, max_evals=100)
    Notes
    -----
    Based on code from https://github.com/hyperopt/hyperopt-sklearn
    """
    def _cost_fn(*args, **kwargs):
        _conn = kwargs.pop('_conn')
        try:
            t_start = time.time()
            rval = cost_fn(*args, **kwargs)
            t_done = time.time()

            if not isinstance(rval, dict):
                rval = dict(loss=rval)
            assert 'loss' in rval, "Returned dictionary must include loss"
            loss = rval['loss']
            assert is_number(loss), "Returned loss must be a number type"
            rval.setdefault('status', hyperopt.STATUS_OK if np.isfinite(loss)
                            else hyperopt.STATUS_FAIL)
            rval.setdefault('duration', t_done - t_start)
            rtype = 'return'

        except Exception as exc:
            rval = exc
            rtype = 'raise'

        # -- return the result to calling process
        _conn.send((rtype, rval))

    def wrapper(*args, **kwargs):
        for k in range(iters):
            conn1, conn2 = Pipe()
            kwargs['_conn'] = conn2
            th = Process(target=_cost_fn, args=args, kwargs=kwargs)
            th.start()
            if conn1.poll(timeout):
                fn_rval = conn1.recv()
                th.join()
            else:
                if verbose >= 1:
                    print("TRIAL TIMED OUT (%d/%d)" % (k+1, iters))
                th.terminate()
                th.join()
                continue

            assert fn_rval[0] in ('raise', 'return')
            if fn_rval[0] == 'raise':
                raise fn_rval[1]
            else:
                return fn_rval[1]

        return {'status': hyperopt.STATUS_FAIL,
                'failure': 'timeout'}

    return wrapper


def wrap_fmin(cost_fn, timeout=None, iters=1, verbose=0):

    def _cost_fn(*args, **kwargs):
        _conn = kwargs.pop('_conn')
        try:
            t_start = time.time()
            rval = cost_fn(*args, **kwargs)
            t_done = time.time()

            if not isinstance(rval, dict):
                rval = dict(loss=rval)
            assert 'loss' in rval, "Returned dictionary must include loss"
            loss = rval['loss']
            assert is_number(loss), "Returned loss must be a number type"
            rval.setdefault('status', hyperopt.STATUS_OK if np.isfinite(loss)
                            else hyperopt.STATUS_FAIL)
            rval.setdefault('duration', t_done - t_start)
            rtype = 'return'

        except Exception as exc:
            rval = exc
            rtype = 'raise'

        # -- return the result to calling process
        _conn.send((rtype, rval))

    def wrapper(*args, **kwargs):
        for k in range(iters):
            conn1, conn2 = Pipe()
            kwargs['_conn'] = conn2
            th = Process(target=_cost_fn, args=args, kwargs=kwargs)
            th.start()
            if conn1.poll(timeout):
                fn_rval = conn1.recv()
                th.join()
            else:
                if verbose >= 1:
                    print("TRIAL TIMED OUT (%d/%d)" % (k+1, iters))
                th.terminate()
                th.join()
                continue

            assert fn_rval[0] in ('raise', 'return')
            if fn_rval[0] == 'raise':
                raise fn_rval[1]
            else:
                return fn_rval[1]

        return {'status': hyperopt.STATUS_FAIL,
                'failure': 'timeout'}

    return wrapper