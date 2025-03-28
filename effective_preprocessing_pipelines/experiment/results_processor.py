from __future__ import print_function
from utils.common import *
from results_processors.pipeline_impact import pipeline_impact
from results_processors.prototype_construction import prototype_construction
from results_processors.experimental_evaluation import exhaustive_prototypes, custom_vs_exhaustive, custom_vs_ml_algorithm
from results_processors.exploratory_analysis import exploratory_analysis
import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("matplotlib.font_manager").disabled = True


def process_results(args):
    """Process the results according to the given parameters.

    Args:
        args: taken from utils.common.parse_args.
    """
    if args.experiment == "pipeline_impact":
        pipeline_impact(args.toy_example, args.cache)
    elif args.experiment == "prototype_construction":
        prototype_construction(args.toy_example, args.cache)
    elif args.experiment == "experimental_evaluation":
        custom_vs_ml_algorithm(args.toy_example, args.cache, plot=True)
        custom_vs_exhaustive(args.toy_example, args.cache, plot=True)
        exhaustive_prototypes(args.toy_example, args.cache, plot=True)
    elif args.experiment == "exploratory_analysis":
        custom_vs_ml_algorithm(args.toy_example, args.cache, plot=False)
        custom_vs_exhaustive(args.toy_example, args.cache, plot=False)
        exhaustive_prototypes(args.toy_example, args.cache, plot=False)
        exploratory_analysis(args.toy_example, args.cache)


if __name__ == "__main__":
    args = parse_args()
    process_results(args)
