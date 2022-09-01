#!/bin/bash

epsfile=$1
pngfile="${epsfile%.*}.png"

convert -colorspace sRGB -density 300 $epsfile -background white -flatten -resize 1863x2636 -units pixelsperinch -density 224.993 $pngfile
