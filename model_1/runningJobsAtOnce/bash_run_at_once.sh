#!/bin/bash

# !!! Change these following two lines according to the model you are running and your dir !!!
code_dir="/home/sl4964/RunPrince/main_model_BS_all.py"
model_type="bpe2char"

#for i in {1,2,3,4};
#do
#    echo $i
#done

learning_rate=(0.00045 0.00012 0.00072 0.00094 0.00032)
embed_dim=(650 826 687 609 821)
dropout=0.3
hidden_size=(994 728 612 682 988)
k_beam_search=(8 15 5 12 16)
n_layers=2
attention=("Bahdanau" "Luong" "Luong" "Bahdanau" "Bahdanau")
experiment="placeholder"
underscore="_"
for (( i=0; i<5; i++));
do
    experiment=$model_type$underscore${learning_rate[$i]}$underscore${embed_dim[$i]}$underscore${hidden_size[$i]}$underscore${k_beam_search[$i]}$underscore${attention[$i]}
    sbatch run_job.sh "$model_type" "${learning_rate[$i]}" "${embed_dim[$i]}" "${hidden_size[$i]}" "${k_beam_search[$i]}" "$n_layers" "$code_dir" "${attention[$i]}" "$experiment"

done
