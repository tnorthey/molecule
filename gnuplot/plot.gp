#!/bin/gnuplot

set term epscairo
set output "out.eps"

set title("Furan Dyson calculations")
set xlabel("Ionisation energy (a.u.)")
set ylabel("Dyson Norm^2 (arb. units)")
set grid

p "dyson.csv" w l t "CAS(6,6)/SVP", "dyson_tzvpp.csv" w l t "CAS(6,6)/TZVPP"
