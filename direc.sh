#! /bin/sh
for filename in $(ls ./numWavfiles) ; do
    str=$filename
    abc=${str:0:1} 
    res="$str,$abc"
    echo $res >> test.csv
done;
