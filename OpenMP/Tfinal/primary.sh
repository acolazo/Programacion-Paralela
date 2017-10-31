#!/bin/bash

#Scripts
#secondary
#secscript
script="secondary.sh"
filename="paralelo"

arg1=(8000)
arg2=(1024 2048 4096)


for i in ${arg1[@]}; do
	for j in ${arg2[@]}; do
		./$script $filename $i $j
	done
done