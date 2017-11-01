#!/bin/bash

#### CAREFEULL WITH DOUBLE QUOTATION IN SPACED DIRECTORY!!!
code_dir="/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Code_Own"
data_dir="/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess"


#### BPE RUNNER.

echo "Learning BPEs from train set..."

if [ ! -f "$data_dir"/train.en.bpe ]; then
    echo "Processing English BPEs..."
    python "$code_dir"/learn_bpe.py -s 20000 < "$data_dir"/train.en.tok > "$data_dir"/train.en.bpe
fi
if [ ! -f "$data_dir"/train.fr.bpe ]; then
    echo "Processing French BPEs..."
    python "$code_dir"/learn_bpe.py -s 20000 < "$data_dir"/train.fr.tok > "$data_dir"/train.fr.bpe
fi

echo "Applying BPEs to train set..."
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.en.bpe < "$data_dir"/train.en.tok > "$data_dir"/train.en.tok.bpe 
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.fr.bpe < "$data_dir"/train.fr.tok > "$data_dir"/train.fr.tok.bpe 

echo "Applying BPEs to valid set..."
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.en.bpe < "$data_dir"/valid.en.tok > "$data_dir"/valid.en.tok.bpe 
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.fr.bpe < "$data_dir"/valid.fr.tok > "$data_dir"/valid.fr.tok.bpe 

echo "Applying BPEs to dev set..."
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.en.bpe < "$data_dir"/dev.en.tok > "$data_dir"/dev.en.tok.bpe 
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.fr.bpe < "$data_dir"/dev.fr.tok > "$data_dir"/dev.fr.tok.bpe 

echo "Applying BPEs to test sets..."
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.en.bpe < "$data_dir"/tst2010.en.tok > "$data_dir"/tst2010.en.tok.bpe 
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.fr.bpe < "$data_dir"/tst2010.fr.tok > "$data_dir"/tst2010.fr.tok.bpe 

python "$code_dir"/apply_bpe.py -c "$data_dir"/train.en.bpe < "$data_dir"/tst2012.en.tok > "$data_dir"/tst2011.en.tok.bpe 
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.fr.bpe < "$data_dir"/tst2012.fr.tok > "$data_dir"/tst2011.fr.tok.bpe 

python "$code_dir"/apply_bpe.py -c "$data_dir"/train.en.bpe < "$data_dir"/tst2012.en.tok > "$data_dir"/tst2012.en.tok.bpe 
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.fr.bpe < "$data_dir"/tst2012.fr.tok > "$data_dir"/tst2012.fr.tok.bpe 

python "$code_dir"/apply_bpe.py -c "$data_dir"/train.en.bpe < "$data_dir"/tst2013.en.tok > "$data_dir"/tst2013.en.tok.bpe 
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.fr.bpe < "$data_dir"/tst2013.fr.tok > "$data_dir"/tst2013.fr.tok.bpe 

python "$code_dir"/apply_bpe.py -c "$data_dir"/train.en.bpe < "$data_dir"/tst2014.en.tok > "$data_dir"/tst2014.en.tok.bpe 
python "$code_dir"/apply_bpe.py -c "$data_dir"/train.fr.bpe < "$data_dir"/tst2014.fr.tok > "$data_dir"/tst2014.fr.tok.bpe 