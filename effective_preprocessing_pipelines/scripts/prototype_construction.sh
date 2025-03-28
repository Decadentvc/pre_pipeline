#!/bin/bash

echo ""
echo -e "\n\n### PROTOTYPE CONSTRUCTION (PC) ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios..."
python experiment/scenario_generator.py -exp prototype_construction $1 $2
echo -e "\tDone.\n"

# EXPERIMENTS
echo -e "PC01. Run experiments with pairs of transformations\n"
## Features Rebalance
echo -e "\tFeatures - Rebalance\n"
python experiment/experiments_launcher.py -exp prototype_construction -mode features_rebalance $1 $2
echo -e "\tRebalance - Features\n"
python experiment/experiments_launcher.py -exp prototype_construction -mode rebalance_features $1 $2
## Features Discretize
echo -e "\tDiscretize - Features\n"
python experiment/experiments_launcher.py -exp prototype_construction -mode discretize_features $1 $2
echo -e "\tFeatures - Discretize\n"
python experiment/experiments_launcher.py -exp prototype_construction -mode features_discretize $1 $2
## Features Normalize
echo -e "\tFeatures - Normalize\n"
python experiment/experiments_launcher.py -exp prototype_construction -mode features_normalize $1 $2
echo -e "\tNormalize - Features\n"
python experiment/experiments_launcher.py -exp prototype_construction -mode normalize_features $1 $2
## Discretize Rebalance
echo -e "\tDiscretize - Rebalance\n"
python experiment/experiments_launcher.py -exp prototype_construction -mode discretize_rebalance $1 $2
echo -e "\tRebalance - Discretize\n"
python experiment/experiments_launcher.py -exp prototype_construction -mode rebalance_discretize $1 $2

# POST-PROCESSING
python experiment/results_processor.py -exp prototype_construction $1 $2