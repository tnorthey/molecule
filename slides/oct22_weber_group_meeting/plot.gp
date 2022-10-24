#!/bin/gnuplot

#set term epscairo
#set output "out.eps"
#set term pngcairo
set terminal pngcairo enhanced font "Times New Roman,20.0" size 1100,720
set output "out.png"

set title("NMM Angle(N-C-O), best fit to experiment x-ray data")
set xlabel("Time (ps)")
set ylabel("Angle (degrees)")
set grid
set xrange [-.25: 4]

p "< paste time.dat geomovie_angle.dat" u ($1+0.2):3 w l lt 2 lw 3 t "Angle(N-C-O)"
