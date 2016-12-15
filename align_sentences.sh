#!/bin/bash

path="models/align_model/non_parallel_corpus/";
trans="trans_"
orig="orig_"
end=".txt"

for i in {1..3720}
do
   yalign-align -a "en" -b "en" "models/align_model/final_model" $path$orig$i$end $path$trans$i$end   
done