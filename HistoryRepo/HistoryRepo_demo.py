#!/usr/bin/env python3
import os
import glob
import json
import csv
import time
import traceback
import io
import sys
import shutil
import contextlib
import threading
import psutil
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from prototype_evalulate_duration_limit_early_stop import ModelManager, EnhancedPipelineRunner


# ----------------- 可配置项 -----------------
DATASETS_DIR = "../datasets/dataset_csv_std_duplicate_removal"
OUTPUT_CSV = "evaluation_results_demo_1.csv"
LOGS_DIR = "runner_logs"
TARGET_COLUMN = "label"
N_TRIALS = 50
CV = 3
TRIAL_TIMEOUT = 60

# candidate-level timeout 秒数
CANDIDATE_TIMEOUT = None

STEPS_ORDER_CANDIDATES = [
    # (1, ['impute', 'encode']),
    (2, ['impute', 'encode', 'normalize', 'features']),
    (3, ['impute', 'encode', 'discretize', 'features']),
    (4, ['impute', 'encode', 'normalize', 'discretize', 'features']),
    (5, ['impute', 'encode', 'discretize', 'features', 'normalize']),
    # (5, ['impute', 'encode', 'normalize', 'features', 'rebalance']),
    # (6, ['impute', 'encode', 'discretize', 'rebalance', 'features']),


    # (1, ['impute', 'encode', 'normalize', 'rebalance', 'features']),
    # (2, ['impute', 'encode', 'normalize', 'features', 'rebalance']),
    # (3, ['impute', 'encode', 'rebalance', 'discretize', 'features']),
    # (4, ['impute', 'encode', 'discretize', 'rebalance', 'features']),
    # (5, ['impute', 'encode', 'discretize', 'features', 'rebalance']),
]
# ------------------------------------------------------

# ----------------- 新增：局部评估控制（在这里设置，按你的要求通过代码内变量控制） -----------------
# 支持 None, int, 或 list[int]
# 注意：如果同时设置 EVALUATE_DATASET_ID 与 EVALUATE_TASK_INDEX，则会报错并退出（互斥）
EVALUATE_DATASET_ID = [30]   # 例如 [3,4] 表示评估排序后的 dataset_paths 中第 3、4 个数据集（1-based）
EVALUATE_TASK_INDEX = None   # 例如 [61, 62] 表示评估全局任务 index 列表（1-based）
# ----------------------------------------------------------------------------------------------------------------


# ---------- 模型参数集合 ----------
def get_param_sets(model_type: str):
    if model_type == "SVM":
        return [
            {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale', 'max_iter': 500},
            # {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto', 'max_iter': 500},
            # {'C': 10.0, 'kernel': 'poly', 'degree': 2, 'gamma': 0.1, 'max_iter': 500},
            # {'C': 0.5, 'kernel': 'sigmoid', 'gamma': 'scale', 'max_iter': 500},
            # {'C': 100.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'auto', 'max_iter': 500}
        ]
    elif model_type == "LR":
        return [
            {'penalty': 'l2', 'C': 0.001, 'solver': 'liblinear', 'multi_class': 'ovr', 'max_iter': 300},

            {'penalty': 'l1', 'C': 0.01, 'solver': 'liblinear', 'multi_class': 'ovr', 'max_iter': 300},
            {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'multi_class': 'multinomial', 'max_iter': 300},
            {'penalty': 'elasticnet', 'C': 1.0, 'solver': 'saga', 'multi_class': 'ovr', 'l1_ratio': 0.5, 'max_iter': 100},
            # {'penalty': None, 'C': 10.0, 'solver': 'newton-cg', 'multi_class': 'multinomial', 'max_iter': 100},
            # {'penalty': 'l2', 'C': 100.0, 'solver': 'sag', 'multi_class': 'ovr', 'max_iter': 100}
        ]
    elif model_type == "RF":
        return [
            {'n_estimators': 50, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt'},
            # {'n_estimators': 100, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2'},
            # {'n_estimators': 200, 'criterion': 'gini', 'max_depth': None, 'max_features': 0.3},
            # {'n_estimators': 150, 'criterion': 'entropy', 'max_depth': 15, 'max_features': 0.5},
            # {'n_estimators': 100, 'criterion': 'gini', 'max_depth': 20, 'max_features': None}
        ]
    elif model_type == "DT":
        return [
            {'criterion': 'gini', 'max_depth': 3, 'max_features': 'sqrt'},
            # {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'log2'},
            # {'criterion': 'gini', 'max_depth': None, 'max_features': 0.5},
            # {'criterion': 'entropy', 'max_depth': 12, 'max_features': None},
            # {'criterion': 'gini', 'max_depth': 5, 'max_features': 0.7}
        ]
    elif model_type == "GBDT":
        return [
            {'learning_rate': 0.01, 'n_estimators': 50, 'max_depth': 3, 'max_features': 'sqrt'},
            # {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 5, 'max_features': 'log2'},
            # {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 7, 'max_features': 0.3},
            # {'learning_rate': 0.2, 'n_estimators': 150, 'max_depth': None, 'max_features': 0.5},
            # {'learning_rate': 0.15, 'n_estimators': 100, 'max_depth': 4, 'max_features': None}
        ]
    elif model_type == "KNN":
        return [
            {'n_neighbors': 3, 'weights': 'distance', 'algorithm': 'kd_tree', 'p': 2, 'metric': 'euclidean'},

            # {'n_neighbors': 3, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2, 'metric': 'minkowski'},
            # {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'kd_tree', 'p': 2, 'metric': 'euclidean'},
            # {'n_neighbors': 7, 'weights': 'uniform', 'algorithm': 'ball_tree', 'p': 1, 'metric': 'manhattan'},
            # {'n_neighbors': 9, 'weights': 'distance', 'algorithm': 'brute', 'p': 2, 'metric': 'minkowski'},
            # {'n_neighbors': 11, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2, 'metric': 'chebyshev'}
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------- 提取准确率 ----------
def extract_accuracy_from_results(results, model_key=None):
    try:
        if isinstance(results, dict):
            if "per_model_best" in results and model_key in results["per_model_best"]:
                r = results["per_model_best"][model_key]
                if isinstance(r, dict) and 'accuracy' in r:
                    return float(r['accuracy'])
            if "optimization_results" in results and isinstance(results["optimization_results"], dict):
                if 'accuracy' in results["optimization_results"]:
                    return float(results["optimization_results"]['accuracy'])
            if "baseline_performance" in results and model_key in results["baseline_performance"]:
                bp = results["baseline_performance"][model_key]
                if isinstance(bp, dict) and 'mean_accuracy' in bp:
                    return float(bp['mean_accuracy'])
    except Exception:
        pass
    return None


# ---------- 安全运行一个 candidate 的评估（带超时） ----------
def run_candidate_with_timeout(runner, model_type, log_path):
    """
    在独立线程中运行 run_runner_silent ，超过 CANDIDATE_TIMEOUT 秒则强制中断
    返回：
        acc, status, results_dict, avg_trial_time_seconds_or_None
    """
    results_container = {}
    thread_ex = None

    def target():
        nonlocal results_container, thread_ex
        try:
            start = time.time()
            results, _, _ = run_runner_silent(runner, True, log_path)
            end = time.time()

            results_container["res"] = results
            results_container["time"] = end - start
        except Exception as e:
            thread_ex = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=CANDIDATE_TIMEOUT)

    if t.is_alive():
        # candidate 卡死，强制终止
        try:
            # 杀掉所有与当前线程相关的僵尸子进程
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                if child.status() == psutil.STATUS_ZOMBIE:
                    child.kill()
        except Exception:
            pass
        # 写入日志
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n--- CANDIDATE TIMEOUT after {CANDIDATE_TIMEOUT}s ---\n")

        # 尝试取 runner 的 best value
        best_acc = None
        try:
            if hasattr(runner, "study") and runner.study is not None and hasattr(runner.study, "best_value"):
                best_acc = runner.study.best_value
        except Exception:
            pass

        # 超时时平均 trial 时长无法可靠计算，返回 None
        return best_acc, "TIMEOUT", None, None

    if thread_ex:
        raise thread_ex
    results = results_container.get("res", None)
    total_elapsed = results_container.get("time", None)

    # --- 计算平均 trial 时长 ---
    trials_executed = None
    try:
        if isinstance(results, dict):
            te_map = results.get("trials_executed", None)
            if isinstance(te_map, dict) and model_type in te_map:
                trials_executed = int(te_map[model_type])
    except Exception:
        trials_executed = None

    # fallback: 如果 runner 对象本身有 n_trials 属性，就用它（通常存在）
    if trials_executed is None:
        try:
            trials_executed = int(getattr(runner, "n_trials", None))
        except Exception:
            trials_executed = None

    avg_trial_time = None
    try:
        if total_elapsed is not None and trials_executed and trials_executed > 0:
            avg_trial_time = float(total_elapsed) / float(trials_executed)
    except Exception:
        avg_trial_time = None

    acc = extract_accuracy_from_results(results, model_type)
    return acc, "SUCCESS", results, avg_trial_time

# ---------- 静默运行 ----------
def run_runner_silent(runner, save_logs=True, log_path=None):
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    results = None
    exc = None
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            results = runner.optimize()
    except Exception as e:
        exc = e
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()
    stdout_buf.close()
    stderr_buf.close()

    if save_logs and log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n--- LOG at {datetime.utcnow().isoformat()} ---\n")
            if out:
                lf.write("STDOUT:\n" + out + "\n")
            if err:
                lf.write("STDERR:\n" + err + "\n")

    if exc:
        exc.captured_stdout = out
        exc.captured_stderr = err
        raise exc

    return results, out, err


# ---------- 主流程 ----------
def main():
    # 保持原有 model 顺序
    # models = ["KNN", "LR", "RF", "SVM", "DT", "GBDT"]
    models = ["LR"]
    dataset_paths = sorted(glob.glob(os.path.join(DATASETS_DIR, "*.csv")))
    if not dataset_paths:
        print(f"ERROR: no datasets found under {DATASETS_DIR}.")
        return

    # 互斥检查
    if EVALUATE_DATASET_ID is not None and EVALUATE_TASK_INDEX is not None:
        print("ERROR: EVALUATE_DATASET_ID 和 EVALUATE_TASK_INDEX 不能同时使用")
        return

    # ---------- 统一解析 dataset_id ----------
    ds_allow_set = None
    if isinstance(EVALUATE_DATASET_ID, int):
        ds_allow_set = {EVALUATE_DATASET_ID}
    elif isinstance(EVALUATE_DATASET_ID, list):
        ds_allow_set = set(EVALUATE_DATASET_ID)

    # ---------- 统一解析 task index ----------
    task_allow_set = None
    if isinstance(EVALUATE_TASK_INDEX, int):
        task_allow_set = {EVALUATE_TASK_INDEX}
    elif isinstance(EVALUATE_TASK_INDEX, list):
        task_allow_set = set(EVALUATE_TASK_INDEX)

    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    n_datasets = len(dataset_paths)
    n_models = len(models)
    n_param_sets = 5
    n_candidates = len(STEPS_ORDER_CANDIDATES)
    tasks_per_dataset = n_models * n_param_sets  
    total_tasks = n_datasets * tasks_per_dataset

    # 决定输出文件名（如果是局部评估则另存）
    global OUTPUT_CSV
    if task_allow_set is not None:
        sorted_idxs = sorted(task_allow_set)
        OUTPUT_CSV = f"evaluation_results_tasks_{'_'.join(map(str, sorted_idxs))}.csv"
    elif ds_allow_set is not None:
        sorted_ds = sorted(ds_allow_set)
        OUTPUT_CSV = f"evaluation_results_datasets_demo_{'_'.join(map(str, sorted_ds))}.csv"
    else:
        OUTPUT_CSV = OUTPUT_CSV  # 保持默认

    # 读取已完成集合（基于输出文件）
    completed = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    completed.add((row["dataset_name"], row["model_type"], int(row["param_set_id"])))
                except Exception:
                    continue

    first_run = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if first_run:
        csv_writer.writerow([
            "index", "timestamp_utc", "dataset_id", "dataset_name",
            "model_type", "param_set_id", "param_json",
            "candidate_accuracies_json", 
            "candidate_times_json",              
            "candidate_times_ranking",
            "candidate_ranking_json",
            "best_accuracy", "best_candidate", "best_candidate_id", "accuracy_ranking",
            "status", "error_message"
        ])
        csv_file.flush()

    mm = ModelManager()
    # 注意：在全量运行中，task_counter 按顺序从 1 增加；在局部运行中我们需要计算正确的全局 index
    task_counter = 0
    start_time = time.time()

    try:
        # 如果用户指定了若干全局 task index（集合），逐条执行并写入同一输出文件，然后退出
        if task_allow_set is not None:
            for idx in sorted(task_allow_set):
                if idx < 1 or idx > total_tasks:
                    print(f"WARNING: requested task index {idx} out of range (1..{total_tasks}), skip.")
                    continue

                # 计算 dataset_id（1-based）、在 dataset 内的偏移（0-based）
                dataset_id = (idx - 1) // tasks_per_dataset + 1
                offset_in_dataset = (idx - 1) % tasks_per_dataset  # 0..29
                model_idx = offset_in_dataset // n_param_sets  # 0..5
                param_set_id = (offset_in_dataset % n_param_sets) + 1  # 1..5
                model_type = models[model_idx]

                # 获取对应的 dataset path
                ds_idx = dataset_id
                if ds_idx < 1 or ds_idx > n_datasets:
                    print(f"WARNING: derived dataset_id {ds_idx} out of range for task {idx}, skip.")
                    continue
                ds_path = dataset_paths[ds_idx - 1]
                ds_name = os.path.basename(ds_path)

                # 设置 task_counter 为该全局 index（保持与全量一致）
                task_counter = idx

                print(f"Running single task by global index {idx} -> dataset_id={dataset_id} ({ds_name}), model={model_type}, param_set_id={param_set_id}")

                # 只跑这一条：构造 param_sets 并取特定 param
                param_sets = get_param_sets(model_type)
                params = param_sets[param_set_id - 1]

                # 执行这一个任务（与原代码块一致）
                candidate_accs = {}
                candidate_times = {}
                ranking_str = ""
                ranking = []
                best_candidate = None
                best_acc = None
                best_cand_id = None
                status = "SUCCESS"
                error_msg = ""

                try:
                    if hasattr(mm, "available_keys") and hasattr(mm, "disable"):
                        for k in mm.available_keys():
                            mm.disable(k)
                    if hasattr(mm, "enable"):
                        mm.enable(model_type)
                    try:
                        mm.set_model_params(model_type, **params)
                    except TypeError:
                        mm.set_model_params(model_type, params)

                    for cand_id, cand in STEPS_ORDER_CANDIDATES:
                        runner = EnhancedPipelineRunner(
                            file_path=ds_path, target_column=TARGET_COLUMN,
                            steps_order_candidates=[cand], model_manager=mm,
                            n_trials=N_TRIALS, cv=CV, trial_timeout=TRIAL_TIMEOUT
                        )
                        log_fname = f"{task_counter:05d}_ds{ds_idx}_model{model_type}_p{param_set_id}_cand{cand_id}.log"
                        log_path = os.path.join(LOGS_DIR, log_fname)

                        try:
                            acc, stat, _, elapsed = run_candidate_with_timeout(runner, model_type, log_path)
                            if stat == "TIMEOUT":
                                status = "CANDIDATE_TIMEOUT"
                        except Exception as e:
                            acc = None
                            error_msg = f"{type(e).__name__}: {str(e)}"
                        candidate_accs[str(cand_id)] = acc
                        candidate_times[str(cand_id)] = elapsed

                    valid_times = [t for t in candidate_times.values() if t is not None]
                    time_dict = {}
                    for cid, t in candidate_times.items():
                        if t is not None:
                            # 把浮点数做个统一化（保留若干小数），避免由于浮点误差导致 grouping 问题
                            # 这里转换为原样字符串 key（保留 6 位小数），分组相等的视为并列
                            key = f"{float(t):.6f}"
                            time_dict.setdefault(key, []).append(cid)

                    # key 从小到大排序
                    sorted_time = sorted(time_dict.items(), key=lambda x: float(x[0]))  # 小→大

                    time_ranking_parts = []
                    for t_key, cids in sorted_time:
                        if len(cids) == 1:
                            time_ranking_parts.append(str(cids[0]))
                        else:
                            time_ranking_parts.append("=".join(sorted(map(str, cids))))

                    time_ranking_str = ">".join(time_ranking_parts)
                    time_ranking = time_ranking_parts
                    # 选最优
                    valid = [(cid, a) for cid, a in candidate_accs.items() if a is not None]
                    valid.sort(key=lambda x: -x[1])
                    if valid:
                        best_cand_id, best_acc = valid[0]
                        for cid, steps in STEPS_ORDER_CANDIDATES:
                            if str(cid) == str(best_cand_id):
                                best_candidate = steps
                                break

                    acc_dict = {}
                    for cid, acc in candidate_accs.items():
                        if acc is not None:
                            acc_dict.setdefault(acc, []).append(cid)
                    sorted_acc = sorted(acc_dict.items(), key=lambda x: -x[0])
                    ranking_parts = []
                    for acc, cids in sorted_acc:
                        if len(cids) == 1:
                            ranking_parts.append(str(cids[0]))
                        else:
                            ranking_parts.append("=".join(sorted(map(str, cids))))
                    ranking_str = ">".join(ranking_parts)
                    ranking = ranking_parts

                except Exception as e:
                    status = "ERROR"
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

                csv_writer.writerow([
                    task_counter, datetime.utcnow().isoformat(), ds_idx, ds_name,
                    model_type, param_set_id, json.dumps(params, ensure_ascii=False),
                    json.dumps(candidate_accs, ensure_ascii=False),
                    json.dumps(candidate_times, ensure_ascii=False),   
                    time_ranking_str,
                    json.dumps(ranking, ensure_ascii=False),
                    best_acc if best_acc is not None else "",
                    json.dumps(best_candidate, ensure_ascii=False) if best_candidate else "",
                    best_cand_id if best_cand_id is not None else "",
                    ranking_str, status, error_msg
                ])
                csv_file.flush()

            print("\nFinished requested task-index set.")
            return  # 已处理完 task 集合，退出主流程

        # 否则进入主循环（可能带 ds_allow_set 或 task_allow_set 为 None）
        else:
            task_counter = 0
            for ds_idx, ds_path in enumerate(dataset_paths, start=1):

                # 若启用 dataset 限制，则过滤掉不需要的 dataset_id
                if ds_allow_set is not None and ds_idx not in ds_allow_set:
                    continue

                ds_name = os.path.basename(ds_path)
                for model_type in models:
                    param_sets = get_param_sets(model_type)
                    for p_idx, params in enumerate(param_sets, start=1):
                        task_counter += 1

                        # 若启用 task index 限制（单个或列表），则仅运行指定 task index
                        if task_allow_set is not None and task_counter not in task_allow_set:
                            continue

                        if (ds_name, model_type, p_idx) in completed:
                            continue

                        candidate_accs = {}
                        candidate_times = {}
                        ranking_str = ""
                        ranking = []
                        best_candidate = None
                        best_acc = None
                        best_cand_id = None
                        status = "SUCCESS"
                        error_msg = ""

                        try:
                            if hasattr(mm, "available_keys") and hasattr(mm, "disable"):
                                for k in mm.available_keys():
                                    mm.disable(k)
                            if hasattr(mm, "enable"):
                                mm.enable(model_type)
                            try:
                                mm.set_model_params(model_type, **params)
                            except TypeError:
                                mm.set_model_params(model_type, params)

                            for cand_id, cand in STEPS_ORDER_CANDIDATES:
                                runner = EnhancedPipelineRunner(
                                    file_path=ds_path, target_column=TARGET_COLUMN,
                                    steps_order_candidates=[cand], model_manager=mm,
                                    n_trials=N_TRIALS, cv=CV, trial_timeout=TRIAL_TIMEOUT
                                )
                                log_fname = f"{task_counter:05d}_ds{ds_idx}_model{model_type}_p{p_idx}_cand{cand_id}.log"
                                log_path = os.path.join(LOGS_DIR, log_fname)

                                try:
                                    acc, stat, _, elapsed = run_candidate_with_timeout(runner, model_type, log_path)
                                    if stat == "TIMEOUT":
                                        status = "CANDIDATE_TIMEOUT"
                                except Exception as e:
                                    acc = None
                                    error_msg = f"{type(e).__name__}: {str(e)}"

                                candidate_accs[str(cand_id)] = acc
                                candidate_times[str(cand_id)] = elapsed

                            valid_times = [t for t in candidate_times.values() if t is not None]
                            time_dict = {}
                            for cid, t in candidate_times.items():
                                if t is not None:
                                    # 把浮点数做个统一化（保留若干小数），避免由于浮点误差导致 grouping 问题
                                    # 这里转换为原样字符串 key（保留 6 位小数），分组相等的视为并列
                                    key = f"{float(t):.6f}"
                                    time_dict.setdefault(key, []).append(cid)

                            # key 从小到大排序
                            sorted_time = sorted(time_dict.items(), key=lambda x: float(x[0]))  # 小→大

                            time_ranking_parts = []
                            for t_key, cids in sorted_time:
                                if len(cids) == 1:
                                    time_ranking_parts.append(str(cids[0]))
                                else:
                                    time_ranking_parts.append("=".join(sorted(map(str, cids))))

                            time_ranking_str = ">".join(time_ranking_parts)
                            time_ranking = time_ranking_parts
                            # 选最优
                            valid = [(cid, a) for cid, a in candidate_accs.items() if a is not None]
                            valid.sort(key=lambda x: -x[1])
                            if valid:
                                best_cand_id, best_acc = valid[0]
                                for cid, steps in STEPS_ORDER_CANDIDATES:
                                    if str(cid) == str(best_cand_id):
                                        best_candidate = steps
                                        break

                            acc_dict = {}
                            for cid, acc in candidate_accs.items():
                                if acc is not None:
                                    acc_dict.setdefault(acc, []).append(cid)
                            sorted_acc = sorted(acc_dict.items(), key=lambda x: -x[0])
                            ranking_parts = []
                            for acc, cids in sorted_acc:
                                if len(cids) == 1:
                                    ranking_parts.append(str(cids[0]))
                                else:
                                    ranking_parts.append("=".join(sorted(map(str, cids))))
                            ranking_str = ">".join(ranking_parts)
                            ranking = ranking_parts

                        except Exception as e:
                            status = "ERROR"
                            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

                        csv_writer.writerow([
                            task_counter, datetime.utcnow().isoformat(), ds_idx, ds_name,
                            model_type, p_idx, json.dumps(params, ensure_ascii=False),
                            json.dumps(candidate_accs, ensure_ascii=False),
                            json.dumps(candidate_times, ensure_ascii=False),   
                            time_ranking_str,
                            json.dumps(ranking, ensure_ascii=False),
                            best_acc if best_acc is not None else "",
                            json.dumps(best_candidate, ensure_ascii=False) if best_candidate else "",
                            best_cand_id if best_cand_id is not None else "",
                            ranking_str, status, error_msg
                        ])
                        csv_file.flush()

            print("\nAll tasks completed (mode finished).")
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
