#!/bin/bash

stepsize=(0.01 0.10 1.00)
gamma=(4.0 8.0)
qmax=(2.0 4.0 8.0 16.0)
noise=(0.00 0.10 0.50)

# start from the template file
cp slides_template.tex slides.tex

# add frames loop
for i in 0 1 2 	# stepsize
do
for j in 0 1	# gamma
do
for l in 0 1 2 	# noise
do
for k in 0 1 2 3	# qmax
do

tmp="stepsize_${stepsize[i]}_gamma_${gamma[j]}_qmax_${qmax[k]}_noise_${noise[l]}"
chi2_graph="nmm_chi2_$tmp.png"
rmsd_graph="nmm_rmsd_$tmp.png"
pcdfit_graph="nmm_pcdfit_$tmp.png"
title="$ q_{\\\\textrm{max}}=${qmax[k]} $ \\\\AA $^{-1}, \\\\eta=${noise[l]}, \\\\Delta s=${stepsize[i]}, \\\\gamma=${gamma[j]}$"

cat frame_template.txt >> slides.tex
sed -i "s:TITLE:$title:" slides.tex
sed -i "s/CHI2_GRAPH/$chi2_graph/" slides.tex
sed -i "s/RMSD_GRAPH/$rmsd_graph/" slides.tex
sed -i "s/PCDFIT_GRAPH/$pcdfit_graph/" slides.tex
echo " " >> slides.tex

done
done
done
done

echo '\end{document}' >> slides.tex

