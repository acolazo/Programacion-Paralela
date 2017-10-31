#!/bin/bash

rm -rf "$PWD/logs/secuencial.csv"
exec &>> "$PWD/logs/secuencial.csv"

filename=(normal ani simd anisimd)


echo -ne "Secuencial";


arg1=(1000 2000 4000)
arg2=(1024 2048 4096)
simd=0

for f in ${filename[@]}; do

	rm -rf $f
	if [ "$f" == "simd" ] ; then
		simd=1
	fi
	if [ $simd -eq 0 ] ; then
		gcc -o $f $f.c -fopenmp -O3
	else
		gcc -o $f $f.c -fopenmp -O3 -mavx
	fi
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