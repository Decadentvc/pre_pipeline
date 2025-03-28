#!/bin/bash

echo ""
echo -e "\n\n### EXPLORATORY ANALYSIS (EA) ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios..."
python experiment/scenario_generator.py -exp pipeline_impact $1 $2
python experiment/scenario_generator.py -exp exhaustive_prototypes $1 $2
python experiment/scenario_generator.py -exp custom_prototypes $1 $2
echo -e "\tDone.\n"

# EXPERIMENTS
echo -e "EA01. SMBO on fixed pre-processing prototypes and ML algorithms\n"
python experiment/experiments_launcher.py -exp pipeline_impact $1 $2

# POST-PROCESSING
echo -e "EA02. Plot pipeline impact\n"
python experiment/results_processor.py -exp pipeline_impact $1 $2

# EXPERIMENTS
echo -e "EA03. SMBO on exhaustive/custom prototypes and ML algorithms\n"
echo -e "\t SMBO on ML algorithms\n"
python experiment/experiments_launcher.py -exp custom_prototypes -mode algorithm $1 $2
echo -e "\tSMBO on custom prototypes and ML algorithms\n"
python experiment/experiments_launcher.py -exp custom_prototypes -mode pipeline_algorithm $1 $2
echo -e "\tSMBO on exhaustive prototypes and ML algorithms\n"
python experiment/experiments_launcher.py -exp exhaustive_prototypes $1 $2

# POST-PROCESSING
python experiment/results_processor.py -exp exploratory_analysis $1 $2