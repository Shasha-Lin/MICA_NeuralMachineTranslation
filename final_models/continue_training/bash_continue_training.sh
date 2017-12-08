#!/bin/bash

# !!! Change these following two lines according to the model you are running and your dir !!!
code_dir="/scratch/mmd378/NLP_2017/newest_branch/final_models/main_model_BS_all_continue.py"
model_type="bpe2char_2"


learning_rate=(0.00087)
dropout=0.3
hidden_size=(657)
k_beam_search=(5)
n_layers=2
attention=("Bahdanau" "Bahdanau" "Bahdanau" "Bahdanau" "Bahdanau")
experiment="placeholder"
underscore="_"
mdif="_cont_"
enc_cp="checkpoints/bpe2char_2_redo__0.00087_657_5_Bahdanau/saved_encoder_1524.pth"
dec_cp="checkpoints/bpe2char_2_redo__0.00087_657_5_Bahdanau/saved_decoder_1524.pth"
epocs=1524
enc_opt_cp="checkpoints/bpe2char_2_redo__0.00087_657_5_Bahdanau/saved_encoder_optim_1524.pth"
dec_opt_cp="checkpoints/bpe2char_2_redo__0.00087_657_5_Bahdanau/saved_decoder_optim_1524.pth"


for (( i=0; i<1; i++));
do
    experiment=$model_type$mdif$underscore${learning_rate[$i]}$underscore${hidden_size[$i]}$underscore${k_beam_search[$i]}$underscore${attention[$i]}
   echo "$model_type" "${learning_rate[$i]}" "${hidden_size[$i]}" "${k_beam_search[$i]}" "$n_layers" "$code_dir" "${attention[$i]}" "$experiment" "$enc_cp" "$dec_cp" "$epocs" 
     sbatch continue_training.sh "$model_type" "${learning_rate[$i]}" "${hidden_size[$i]}" "${k_beam_search[$i]}" "$n_layers" "$code_dir" "${attention[$i]}" "$experiment" "$enc_cp" "$dec_cp" "$epocs"

done
