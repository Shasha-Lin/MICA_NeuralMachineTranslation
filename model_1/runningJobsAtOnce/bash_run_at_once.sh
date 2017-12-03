#!/bin/bash

### code dir contains this file and the file_merger code
code_dir="/home/ak6201/RunPrince/main_model_BS_all.py"

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



for (( i=0; i<5; i++));
do

    sbatch run_all_jobs.sh "bpe2bpe" "${learning_rate[$i]}" "${embed_dim[$i]}" "${hidden_size[$i]}" "${k_beam_search[$i]}" "$n_layers" "$code_dir"

done
