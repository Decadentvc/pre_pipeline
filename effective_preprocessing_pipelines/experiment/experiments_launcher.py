from utils import serializer
from utils.common import *
from utils import scenarios as scenarios_util
from utils.auto_pipeline_builder import framework_table_pipelines, pseudo_exhaustive_pipelines
from tqdm import tqdm
from prettytable import PrettyTable
from six import iteritems
from functools import reduce
import os
import json
import shutil
import warnings
import yaml
import psutil
import subprocess
import datetime

warnings.filterwarnings("ignore")

GLOBAL_SEED = 42


def kill(proc_pid):
    """Kill a process given its process id.

    Args:
        proc_pid: process id.
    """
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def run_cmd(cmd, current_scenario, result_path, stdout_path, stderr_path, toy):
    """Launch a proccess that computes the experiment by running a specific command on the CLI.

    Args:
        cmd: command to run.
        current_scenario: scenario of the experiment.
        result_path: where to save the results.
        stdout_path: where to save the standard output.
        stderr_path: where to save the standard error.
        toy: wheter it is the toy example or not.
    """
    open(stdout_path, "w")
    open(stderr_path, "w")
    with open(stdout_path, "a") as log_out:
        with open(stderr_path, "a") as log_err:
            max_time = 50 if toy else 1000
            try:
                process = subprocess.Popen(
                    cmd, shell=True, stdout=log_out, stderr=log_err
                )
                process.wait(timeout=max_time)
            except Exception as e:
                kill(process.pid)
                serializer.serialize_results(
                    scenario=current_scenario, result_path=result_path
                )


def launch_experiments(args):
    """Launch the experiments according to the given parameters.

    Args:
        args: taken from utils.common.parse_args.
    """

    
    print("\t\tChecking scenarios with results...\n")
    
    # Get scenario and result path
    if args.toy_example == True:
        scenario_path = create_directory(SCENARIO_PATH, "toy")
        result_path = create_directory(RAW_RESULT_PATH, "toy")
    elif args.cache == True:
        scenario_path = create_directory(SCENARIO_PATH, "paper")
        result_path = create_directory(RAW_RESULT_PATH, "paper")
    else:
        scenario_path = create_directory(SCENARIO_PATH, "paper_new")
        result_path = create_directory(RAW_RESULT_PATH, "paper_new")

    scenario_path = create_directory(scenario_path, args.experiment)
    result_path = create_directory(result_path, args.experiment)

    # Create directories for saving the results
    if args.mode:
        if args.mode not in ["algorithm", "pipeline_algorithm"]:
            pipeline = args.mode.split("_")
            if pipeline[0] <= pipeline[1]:
                result_path = create_directory(result_path, args.mode)
                result_path = create_directory(result_path, "conf1")
            else:
                result_path = create_directory(result_path, "_".join(sorted(pipeline)))
                result_path = create_directory(result_path, "conf2")
        else:
            scenario_path = create_directory(scenario_path, args.mode)
            result_path = create_directory(result_path, args.mode)

    # Gather list of scenarios
    scenario_list = [p for p in os.listdir(scenario_path) if ".yaml" in p]
    result_list = [p for p in os.listdir(result_path) if ".json" in p]
    scenarios = {}

    # Determine which one have no result files
    for scenario in scenario_list:
        base_scenario = scenario.split(".yaml")[0]
        if scenario not in scenarios:
            scenarios[scenario] = {"raw_results": None, "path": scenario}
        for result in result_list:
            base_result = result.split(".json")[0]
            if base_result.__eq__(base_scenario):
                scenarios[scenario]["raw_results"] = result

    # Calculate total amount of time
    total_runtime = 0
    for path, scenario in iteritems(scenarios):
        with open(os.path.join(scenario_path, path), "r") as f:
            details = None
            try:
                details = yaml.safe_load(f)
            except Exception:
                details = None
                scenario["status"] = "Invalid YAML"
            if details is not None:
                try:
                    runtime = details["setup"]["runtime"]
                    scenario["status"] = "Ok"
                    if args.experiment == "exhaustive_prototypes":
                        runtime *= 24
                    if args.experiment == "custom_prototypes" and args.mode == "pipeline_algorithm":
                        runtime *= 4
                    scenario["runtime"] = runtime
                    if scenario["raw_results"] is None:
                        total_runtime += runtime
                except:
                    scenario["status"] = "No runtime info"

    # Display list of scenario to be run
    invalid_scenarios = {k: v for k, v in iteritems(scenarios) if v["status"] != "Ok"}
    t_invalid = PrettyTable(["PATH", "STATUS"])
    t_invalid.align["PATH"] = "l"
    for v in invalid_scenarios.values():
        t_invalid.add_row([v["path"], v["status"]])

    scenario_with_results = {
        k: v
        for k, v in iteritems(scenarios)
        if v["status"] == "Ok" and v["raw_results"] is not None
    }
    t_with_results = PrettyTable(["PATH", "RUNTIME", "STATUS", "RESULTS"])
    t_with_results.align["PATH"] = "l"
    t_with_results.align["RESULTS"] = "l"
    for v in scenario_with_results.values():
        t_with_results.add_row(
            [v["path"], str(v["runtime"]) + "s", v["status"], v["raw_results"]]
        )

    to_run = {
        k: v
        for k, v in iteritems(scenarios)
        if v["status"] == "Ok" and v["raw_results"] is None
    }
    t_to_run = PrettyTable(["PATH", "RUNTIME", "STATUS"])
    t_to_run.align["PATH"] = "l"
    for v in to_run.values():
        t_to_run.add_row([v["path"], str(v["runtime"]) + "s", v["status"]])

    print(f"\t\tnum invalid scenarios: {len(invalid_scenarios)}")
    print(f"\t\tnum scenarios with results: {len(scenario_with_results)}")
    print(f"\t\tnum scenarios to run: {len(to_run)}")
    if len(to_run) > 0:
        print(
            "\t\t\testimated time: {} ({}s)\n".format(
                datetime.timedelta(seconds=total_runtime * 2), total_runtime * 2
            )
        )

    # Run experiment one by one
    if to_run.values():
        with tqdm(total=len(to_run)) as pbar:
            for info in to_run.values():

                # Define the common data structures
                base_scenario = info["path"].split(".yaml")[0]
                current_scenario_path = os.path.join(scenario_path, info["path"])
                current_scenario = scenarios_util.load(current_scenario_path)

                # Run experiments regarding the PC and EA workflows
                if args.experiment == "prototype_construction" or args.experiment == "pipeline_impact":

                    # Define the pipeline to optimize
                    if args.experiment == "prototype_construction":
                        pipeline = args.mode.split("_")
                    else:
                        if base_scenario.startswith("knn"):
                            pipeline = ["impute", "encode", "normalize", "rebalance", "features"]
                        elif base_scenario.startswith("nb"):
                            pipeline = ["impute", "encode", "normalize", "features", "rebalance"]
                        else:
                            pipeline = ["impute", "encode", "normalize", "rebalance", "features"]

                    # Create the command
                    cmd = "python experiment/main.py -s {} -c control.seed={} -p {} -r {} -exp {}".format(
                        current_scenario_path,
                        GLOBAL_SEED,
                        reduce(lambda x, y: x + " " + y, pipeline),
                        result_path,
                        args.experiment,
                    )
                    if args.toy_example:
                        cmd += " -toy true"

                    # Set stdout and stderr
                    stdout_path = os.path.join(result_path, "{}_stdout.txt".format(base_scenario))
                    stderr_path = os.path.join(result_path, "{}_stderr.txt".format(base_scenario))

                    # Launch the experiments
                    run_cmd(
                        cmd,
                        current_scenario,
                        result_path,
                        stdout_path,
                        stderr_path,
                        args.toy_example,
                    )

                # Run experiments regarding the EE workflow (exhaustive prototypes + ML algorithm)
                elif args.experiment == "exhaustive_prototypes":

                    # Get the exhaustive prototypes
                    pipelines = framework_table_pipelines()

                    data_to_write = {}
                    data_to_write["pipelines"] = []
                    results = []

                    # For each prototype...
                    for i in range(0, len(pipelines)):
                        pipeline = pipelines[i]
                        cmd = "python experiment/main.py -s {} -c control.seed={} -p {} -r {} -np {} -exp {}".format(
                            current_scenario_path,
                            GLOBAL_SEED,
                            pipeline,
                            result_path,
                            len(pipelines),
                            args.experiment,
                        )
                        if args.toy_example:
                            cmd += " -toy true"

                        # Set stdout and stderr
                        stdout_path = os.path.join(result_path, "{}_{}_stdout.txt".format(base_scenario, str(i)))
                        stderr_path = os.path.join(result_path, "{}_{}_stderr.txt".format(base_scenario, str(i)))

                        # Launch the experiments
                        run_cmd(
                            cmd,
                            current_scenario,
                            result_path,
                            stdout_path,
                            stderr_path,
                            args.toy_example,
                        )

                        # Rename the output
                        try:
                            os.rename(
                                os.path.join(result_path, "{}.json".format(base_scenario)),
                                os.path.join(result_path, "{}.json".format(base_scenario + "_" + str(i)))
                            )
                            with open(
                                os.path.join(result_path, "{}.json".format(base_scenario + "_" + str(i)))
                            ) as json_file:
                                data = json.load(json_file)
                                accuracy = (data["context"]["best_config"]["score"] // 0.0001 / 100)
                                results.append(accuracy)
                        except Exception:
                            accuracy = 0

                        # Store the accuracy
                        data_to_write["pipelines"].append(
                            {
                                "index": str(i),
                                "pipeline": pipeline,
                                "accuracy": accuracy,
                            }
                        )

                    # Store the best pipeline
                    try:
                        with open(
                            os.path.join(result_path, "{}.json".format(base_scenario)), "w"
                        ) as outfile:
                            json.dump(data_to_write, outfile)
                    except:
                        print("I didn't manage to write")

                # Run the experiments of the EE workflow (custom prototypes + ML algorithm)
                elif args.experiment == "custom_prototypes":

                    if args.mode == "pipeline_algorithm":

                        # Get the effective prototypes
                        pipelines = pseudo_exhaustive_pipelines()
                        results = []

                        # For each prototype...
                        for i in range(0, len(pipelines)):
                            pipeline = pipelines[i]
                            cmd = "python experiment/main.py -s {} -c control.seed={} -p {} -r {} -m {} -np {} -exp {}".format(
                                current_scenario_path,
                                GLOBAL_SEED,
                                pipeline,
                                result_path,
                                "pipeline_algorithm",
                                len(pipelines),
                                args.experiment,
                            )
                            if args.toy_example:
                                cmd += " -toy true"

                            # Set stdout and stderr
                            stdout_path = os.path.join(result_path, "{}_{}_stdout.txt".format(base_scenario, str(i)))
                            stderr_path = os.path.join(result_path, "{}_{}_stderr.txt".format(base_scenario, str(i)))

                            # Launch the experiments
                            run_cmd(
                                cmd,
                                current_scenario,
                                result_path,
                                stdout_path,
                                stderr_path,
                                args.toy_example,
                            )

                            # Rename the output files
                            try:
                                os.rename(
                                    os.path.join(result_path, "{}.json".format(base_scenario)),
                                    os.path.join(result_path, "{}_{}.json".format(base_scenario, str(i))),
                                )
                                with open(
                                    os.path.join(result_path, "{}_{}.json".format(base_scenario, str(i)))
                                ) as json_file:
                                    data = json.load(json_file)
                                    accuracy = (data["context"]["best_config"]["score"] // 0.0001 / 100)
                                    results.append(accuracy)
                            except:
                                accuracy = 0
                                results.append(accuracy)

                        # Store the best performance
                        try:
                            max_i = 0
                            for i in range(1, len(pipelines)):
                                if results[i] > results[max_i]:
                                    max_i = i
                            src_dir = os.path.join(result_path, "{}.json".format(base_scenario + "_" + str(max_i)))
                            dst_dir = os.path.join(result_path, "{}.json".format(base_scenario + "_best_pipeline"))
                            shutil.copy(src_dir, dst_dir)
                        except:
                            with open(
                                os.path.join(result_path, "{}.txt".format(base_scenario + "_best_pipeline")), "a"
                            ) as log_out:
                                log_out.write("trying to get the best pipeline: no available result")

                        # Run the algorithm optimization with the best pipeline
                        try:
                            with open(
                                os.path.join(result_path,"{}.json".format(base_scenario + "_best_pipeline"))
                            ) as json_file:
                                data = json.load(json_file)
                                pipeline = data["pipeline"]
                            cmd = "python experiment/main.py -s {} -c control.seed={} -p {} -r {} -m {} -np {} -exp {}".format(
                                current_scenario_path,
                                GLOBAL_SEED,
                                " ".join(pipeline),
                                result_path,
                                "algorithm",
                                len(pipelines),
                                args.experiment,
                            )
                            if args.toy_example:
                                cmd += " -toy true"

                            # Set stdout and stderr
                            stdout_path = os.path.join(result_path, "{}_stdout.txt".format(base_scenario))
                            stderr_path = os.path.join(result_path, "{}_stderr.txt".format(base_scenario))

                            # Launch the experiments
                            run_cmd(
                                cmd,
                                current_scenario,
                                result_path,
                                stdout_path,
                                stderr_path,
                                args.toy_example,
                            )

                        except:
                            with open(
                                os.path.join(result_path, "{}.txt".format(base_scenario)), "a"
                            ) as log_out:
                                log_out.write("\ntrying to run best pipeline and algorithm: could not find a pipeline")

                    # Run the experiments of the EE workflow (Just ML algorithm)
                    elif args.mode == "algorithm":
                        cmd = "python experiment/main.py -s {} -c control.seed={} -p {} -r {} -m {} -np {} -exp {}".format(
                            current_scenario_path,
                            GLOBAL_SEED,
                            "impute encode",
                            result_path,
                            "algorithm",
                            0,
                            args.experiment,
                        )
                        if args.toy_example:
                            cmd += " -toy true"

                        # Set stdout and stderr
                        stdout_path = os.path.join(result_path, "{}_stdout.txt".format(base_scenario))
                        stderr_path = os.path.join(result_path, "{}_stderr.txt".format(base_scenario))
                        
                        # Launch the experiments
                        run_cmd(
                            cmd,
                            current_scenario,
                            result_path,
                            stdout_path,
                            stderr_path,
                            args.toy_example,
                        )

                    else:
                        raise Exception("unvalid mode option")
                else:
                    raise Exception("unvalid experiment option")
                pbar.update()
    
    print()

if __name__ == "__main__":
    args = parse_args()
    launch_experiments(args)
