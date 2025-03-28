#!/bin/bash

echo ""
echo -e "\n\n### EXPERIMENTAL EVALUATION (EE) ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios..."
python experiment/scenario_generator.py -exp exhaustive_prototypes $1 $2
python experiment/scenario_generator.py -exp custom_prototypes $1 $2
echo -e "\tDone.\n"

# EXPERIMENTS
echo -e "EE01. SMBO on ML algorithms\n"
python experiment/experiments_launcher.py -exp custom_prototypes -mode algorithm $1 $2
echo -e "EE02-EE04. SMBO on effective pre-processing prototypes and ML algorithms\n"
python experiment/experiments_launcher.py -exp custom_prototypes -mode pipeline_algorithm $1 $2
echo -e "EE05-EE07. SMBO on exhaustive pre-processing prototypes and ML algorithms\n"
python experiment/experiments_launcher.py -exp exhaustive_prototypes $1 $2

# POST-PROCESSING
python experiment/results_processor.py -exp experimental_evaluation $1 $2
