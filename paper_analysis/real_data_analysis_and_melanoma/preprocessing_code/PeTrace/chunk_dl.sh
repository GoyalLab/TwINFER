#!/bin/bash
# usage: chunk_dl.sh <name> <idx> <start> <end>   (inclusive byte range; resumable per chunk)
name=$1; idx=$2; start=$3; end=$4
p=parts/${name}.$(printf %04d $idx)
want=$((end-start+1))
for try in $(seq 1 30); do
  have=$( [ -f $p ] && stat -c%s $p || echo 0 )
  [ "$have" -ge "$want" ] && exit 0
  curl -sS -L -r $((start+have))-$end "https://zenodo.org/records/23088914/files/${name}?download=1" >> $p || sleep 5
done
exit 1
