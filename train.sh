#!/bin/bash
# A simple shell script for running in training mode
# defaults.py file contains the experimental variables

./main.py --handle=train --device=cpu --no-screen-display --save-model-at-termination boxing
