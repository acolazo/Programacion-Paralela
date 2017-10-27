#!/bin/bash

#Execute as ./script FILE_TO_EXECUTE ARGUMENT1 ARGUMENT2
rm -rf "$PWD/logs/$2-$3.csv"
exec &>> "$PWD/logs/$2-$3.csv"
#http://tldp.org/LDP/Bash-Beginners-Guide/html/sect_07_03.html
#print variable on a screen

#VARIABLES PARA MODIFICAR
custom1="{1}, {4}, {2}, {5}, {3}, {6}, {7}, {10}, {8}, {11}, {9}, {12}"
custom2="{1}, {4}, {7}, {10}, {2}, {5}, {8}, {11}, {3}, {6}, {9}, {12}"
CUANTO_VECES_EJECUTO=7
CON_CUANTOS_THREADS_EMPIEZO=2
CUANTAS_VECES_AUMENTO_THREADS=6
places=(cores cores threads threads $custom1 $custom2)
binding=(close spread close false true true)

echo "$2-$3"

#Variable Afinidad - Distintas afinidades que se utilizan


contador=0
for p in ${places[@]}; do

	echo -ne "PLACES:";
	echo -ne $p;
	echo -ne "|BIND:";
	echo -ne ${binding[contador]};

	export OMP_PLACES=$p
	export OMP_PROC_BIND=${binding[contador]}

#Variable Flags - Distintos Flags que se usan
FLAGS=4
while [ $FLAGS -gt 0 ]; do
	#Start Loop2
	rm -rf $1
	if [ $FLAGS -eq 4 ] ; then
		echo -ne ",NO OPT-NO MAVX";
		gcc -o $1 $1.c -fopenmp
	fi
	if [ $FLAGS -eq 3 ] ; then
		echo -ne ",NO OPT-MAVX";
		gcc -o $1 $1.c -fopenmp -mavx
	fi
	if [ $FLAGS -eq 2 ] ; then
		echo -ne ",OPT-NO MAVX";
		gcc -o $1 $1.c -fopenmp -O3
	fi
	if [ $FLAGS -eq 1 ] ; then
		echo -ne ",OPT-MAVX";
		gcc -o $1 $1.c -fopenmp -mavx -O3
	fi
	#Variable COUNT_THREADS - Cuantas veces se duplica
	COUNT_THREADS=$CUANTAS_VECES_DUPLICO_THREADS
	#Variable Threads - Con cuantos Threads comienza.
	THREADS=$CON_CUANTOS_THREADS_EMPIEZO
	FIRST=1
	while [ $COUNT_THREADS -gt 0 ]; do
		#Start Loop3
		if [ $FIRST -eq 1 ]; then
			echo -ne ",$THREADS threads,";
			let FIRST=FIRST-1
		else
			echo -ne ",,$THREADS threads,";
		fi
		export OMP_NUM_THREADS=$THREADS

		#Variable Count - Cantidad de veces que se ejecuta un programa.
		COUNT=$CUANTO_VECES_EJECUTO
		while [ $COUNT -gt 0 ]; do
			#Start Loop4

	#echo Value of count is: $COUNT
	./$1 $2 $3
	#./FILENAME ARGUMENT1 ARGUMENT2
	#./ani > allout.txt 2>&1
	let COUNT=COUNT-1
done 
#End Loop4
echo ""

let THREADS=THREADS+2
let COUNT_THREADS=COUNT_THREADS-1
done
#End Loop 3

let FLAGS=FLAGS-1
done
#End Loop 2

let contador=contador+1
done
#End Loop1

echo ""