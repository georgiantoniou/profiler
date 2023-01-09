#!/bin/bash

export ANSIBLE_HOST_KEY_CHECKING=False
export PROFILER_HOME=/tmp

if [ -z $1 ]; then
  echo "wrong arguments pass number of nodes to monitor and range"
  echo "e.g ./setHosts.sh 2 2 4"
  exit
else
  start=$3
  end=$4
fi

echo "[profiler]" > hosts
for i in [$start..$end];
do
  echo "node$i" >> hosts
done

