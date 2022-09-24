#!/bin/gnuplot

#set term epscairo
#set output "out.eps"
#set term pngcairo
set terminal pngcairo enhanced font "Times New Roman,20.0" size 1280,720
set output "out.png"

set title("NMM fit N-C-O angle")
set xlabel("Time (ps)")
set ylabel("Angle (degrees)")
set grid
set xrange [-.4: 12]

p "< paste time.dat geomovie_angle.dat" u 1:3 w l lt 2 lw 3 t "Angle(N-C-O)"
