#! /bin/bash
cd ~/Desktop/rcars/rc
pyenv activate rcars
if [ -z "$1" ]
  then
    echo "No argument supplied"
  else
    echo "Starting python script"
    python run_$1.py
  fi
