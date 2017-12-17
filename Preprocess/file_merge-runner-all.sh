#!/bin/bash

### code dir contains this file and the file_merger code 
code_dir="/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/MICA_NeuralMachineTranslation/Preprocess"

### data dir contains all bpe and tokenized files
data_dir="/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess"

### folder to contain output folders (Model1_ready or Model2_ready)
data_output_folder="ModelCharzed2"


for i in "train" "dev" "valid"
do
        #bpe2char
        python "$code_dir"/file_merger.py --file1 "$data_dir/"$i".en.tok.bpe" \
                                  --file2 "$data_dir/"$i".fr.norm.char" \
                                  --outdir "$data_dir/$data_output_folder/"$i \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".bpe2char"
        #bpe2bpe
        python "$code_dir"/file_merger.py --file1  "$data_dir/"$i".en.tok.bpe"\
                                  --file2 "$data_dir/"$i".fr.tok.bpe" \
                                  --outdir "$data_dir/$data_output_folder/"$i \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".bpe2bpe"
        #seq2seq
        python "$code_dir"/file_merger.py --file1  "$data_dir/"$i".en.tok"\
                                  --file2 "$data_dir/"$i".fr.tok" \
                                  --outdir "$data_dir/$data_output_folder/"$i \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".seq2seq"
done



for i in "tst2010" "tst2011" "tst2012" "tst2013" "tst2014"
do
        #bpe2char
        python "$code_dir"/file_merger.py --file1 "$data_dir/"$i".en.tok.bpe" \
                                  --file2 "$data_dir/"$i".fr.norm.char" \
                                  --outdir "$data_dir/$data_output_folder/test" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term "."$i"-bpe2char"
        #bpe2bpe
        python "$code_dir"/file_merger.py --file1  "$data_dir/"$i".en.tok.bpe"\
                                  --file2 "$data_dir/"$i".fr.tok.bpe" \
                                  --outdir "$data_dir/$data_output_folder/test" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term "."$i"-bpe2bpe"
        #seq2seq
        python "$code_dir"/file_merger.py --file1  "$data_dir/"$i".en.tok"\
                                  --file2 "$data_dir/"$i".fr.tok" \
                                  --outdir "$data_dir/$data_output_folder/test" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term "."$i"-seq2seq"
done





