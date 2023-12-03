#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

start_time=$SECONDS

echo -en "\n"
echo "#################"
echo "    COMPILING    "
echo "#################"
echo -en "\n"

make _net

echo -en "\n"
echo "#################"
echo "     RUNNING     "
echo "################"
echo -en "\n"


[ -w out ] && stdbuf -oL nice -n 19 ./build/net 2>&1 | tee out || ./build/net

if { [ -e out ] && [ -w out ]; } || { touch out && [ -w out ]; }; then
    stdbuf -oL nice -n 19 ./build/net 2>&1 | tee out
else
    ./build/net
fi


elapsed_time=$((SECONDS - start_time))
echo -en "Elapsed time: $elapsed_time seconds\n" | tee -a out

cat out > /home/x548309/out
