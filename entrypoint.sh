#!/bin/bash
set -e

if [ "$1" = "default" ]; then
    # do default things
    # echo "Running with default settings"
    exec conda run -n $PYTHONENV jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --NotebookApp.password=''
    # exec 'bin/bash'

else
    echo "Running user supplied args"
    # if the user supplied i.e. /bin/bash
    exec "$@"
fi