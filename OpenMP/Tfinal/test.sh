#!/bin/bash

#Execute as ./script FILE_TO_EXECUTE ARGUMENT1 ARGUMENT2

#http://tldp.org/LDP/Bash-Beginners-Guide/html/sect_07_03.html
#print variable on a screen

#VARIABLES PARA MODIFICAR
custom1="{1}, {4}, {2}, {5}, {3}, {6}, {7}, {10}, {8}, {11}, {9}, {12}"
custom2="{1}, {4}, {7}, {10}, {2}, {5}, {8}, {11}, {3}, {6}, {9}, {12}"
custom3="{0}, {1}, {2}, {3}"

CUANTO_VECES_EJECUTO=1 #7
CON_CUANTOS_THREADS_EMPIEZO=2
CUANTAS_VECES_AUMENTO_THREADS=1 #6
declare -a places=("cores" "cores" "threads" "threads" "$custom1" "$custom2")
binding=(close spread close false true true)

echo "$2-$3"

#Variable Afinidad - Distintas afinidades que se utilizan


contador=0
for p in "${places[@]}"; do

	echo -ne "PLACES:";
	echo -ne "$p";
	echo -ne "|BIND:";
	echo -ne ${binding[contador]};
	echo ""
	export OMP_PLACES="$p"
	export | grep 'OMP_PLACES'
done
