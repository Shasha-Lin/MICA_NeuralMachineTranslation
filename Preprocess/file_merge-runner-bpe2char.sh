#!/bin/bash

code_dir="/Users/millie/Documents/NLP_fall_2017/project/Preprocess_copy"

python "$code_dir"/file_merger.py --file1 "/Users/millie/Documents/NLP_fall_2017/project/en-fr/train.en.tok.bpe" \
                                  --file2 "/Users/millie/Documents/NLP_fall_2017/project/en-fr/train.fr.tok" \
                                  --outdir "/Users/millie/Documents/NLP_fall_2017/project/Model2_ready/train" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".bpe2char"
                                  
