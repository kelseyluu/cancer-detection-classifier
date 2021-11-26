#!/bin/bash
while read A B C D
do
echo fixedStep chrom=$A start=$(($B+1)) step=$(($C-$B)) span=$(($C-$B))
echo $D
done
