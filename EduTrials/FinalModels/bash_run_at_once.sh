#!/bin/bash

# !!! Change these following two lines according to the model you are running and your dir !!!
code_dir="./train.py"
model_type="bpe2bpe"


learning_rate=(0.000451 0.000121 0.000721 0.000941 0.000321)
dropout=0.3
hidden_size=(993 729 613 683 9889)
k_beam_search=(8 15 5 12 16)
n_layers=2
attention=("Bahdanau" "Luong" "Luong" "Bahdanau" "Bahdanau")
experiment="placeholder"
underscore="_"
for (( i=0; i<5; i++));
do
    experiment=$model_type$underscore${learning_rate[$i]}$underscore${hidden_size[$i]}$underscore${k_beam_search[$i]}$underscore${attention[$i]}
    
    sbatch run_job.sh "$model_type" "${learning_rate[$i]}" "${hidden_size[$i]}" "${k_beam_search[$i]}" "$n_layers" "$code_dir" "${attention[$i]}" "$experiment"

done
