#!/bin/bash
[ "$(ls -A /home/autoprep)" ] || cp -R /home/dump/. /home/autoprep

cd /home/autoprep
chmod 777 ./scripts/*

./scripts/exploratory_analysis.sh $1 $2

./scripts/prototype_construction.sh $1 $2

./scripts/experimental_evaluation.sh $1 $2