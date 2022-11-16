#! /bin/bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
 eval "$(pyenv init -)"
fi
eval "$(pyenv virtualenv-init -)"
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}:~/bin
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRA>


cd ~/Desktop/rcars/rc
pyenv activate rcars
if [ -z "$1" ]
  then
    echo "No argument supplied"
  else
    echo "Starting python script"
    python run_$1.py
  fi
