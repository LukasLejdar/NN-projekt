#!/bin/bash

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

setterm -linewrap off
stdbuf -oL nice -n 19 ./build/net 2>&1 | tee out

elapsed_time=$((SECONDS - start_time))
echo -en "Elapsed time: $elapsed_time seconds\n" | tee -a out

cat out > /home/x548309/out
