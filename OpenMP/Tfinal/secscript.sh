#!/bin/bash

rm -rf "$PWD/logs/secuencial.csv"
exec &>> "$PWD/logs/secuencial.csv"

filename=(normal ani simd anisimd)


echo -ne "Secuencial";


arg1=(1000 2000)
arg2=(128 256 512 1024)


for f in ${filename[@]}; do

	rm -rf $f
	gcc -o $f $f.c -fopenmp -mavx -O3
	echo "";
	echo -ne $f;
	for i in ${arg1[@]}; do
		for j in ${arg2[@]}; do
			echo ""
			echo -ne "M:$i|B:$j,";
		#Variable Count
		COUNT=5
		while [ $COUNT -gt 0 ]; do
			./$f $i $j
			let COUNT=COUNT-1
		done 
	done
done
done