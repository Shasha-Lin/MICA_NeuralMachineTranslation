#!/bin/bash

code_dir="/Users/millie/Documents/NLP_fall_2017/project/Preprocess_copy"
data_dir="/Users/millie/Documents/NLP_fall_2017/project"



for i in "train" "dev" "test"
do
        python "$code_dir"/file_merger.py --file1 "$data_dir/en-fr/"$i".en.tok.bpe" \
                                  --file2 "$data_dir/en-fr/"$i".fr.tok" \
                                  --outdir "$data_dir/Model2_ready/"$i \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".bpe2char"
done



for i in "tst2010" "tst2011" "tst2012" "tst2013" "tst2014"
do
        python "$code_dir"/file_merger.py --file1 "$data_dir/en-fr/"$i".en.tok.bpe" \
                                  --file2 "$data_dir/en-fr/"$i".fr.tok" \
                                  --outdir "$data_dir/Model2_ready/"$i \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term "."$i"-bpe2char"
done



