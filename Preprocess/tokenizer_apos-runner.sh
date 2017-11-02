#!/bin/bash

#### CAREFEULL WITH DOUBLE QUOTATION IN SPACED DIRECTORY!!!
code_dir="/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Code_Own"
data_dir="/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess"


#### Tokenizer_apos Runner.

# perl $P1/tokenizer_apos.perl -threads 5 -l $S < all_${S}-${T}.${S}.norm > all_${S}-${T}.${S}.tok 

echo "Tokenizing validation set"
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "en" < "$data_dir"/valid.en.norm > "$data_dir"/valid.en.tok
echo "..."
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "fr" < "$data_dir"/valid.fr.norm > "$data_dir"/valid.fr.tok

echo "Tokenizing train set"
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "en" < "$data_dir"/train.en.norm > "$data_dir"/train.en.tok
echo "..."
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "fr" < "$data_dir"/train.fr.norm > "$data_dir"/train.fr.tok

echo "Tokenizing dev set"
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "en" < "$data_dir"/dev.en.norm > "$data_dir"/dev.en.tok
echo "..."
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "fr" < "$data_dir"/dev.fr.norm > "$data_dir"/dev.fr.tok

echo "Tokenizing test set (2010)"
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "en" < "$data_dir"/tst2010.en.norm > "$data_dir"/tst2010.en.tok 
echo "..."
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "fr" < "$data_dir"/tst2010.fr.norm> "$data_dir"/tst2010.fr.tok 

echo "Tokenizing test set (2011)"
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "en" < "$data_dir"/tst2011.en.norm > "$data_dir"/tst2011.en.tok 
echo "..."
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "fr" < "$data_dir"/tst2011.fr.norm> "$data_dir"/tst2011.fr.tok 

echo "Tokenizing test set (2012)"
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "en" < "$data_dir"/tst2012.en.norm > "$data_dir"/tst2012.en.tok 
echo "..."
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "fr" < "$data_dir"/tst2012.fr.norm> "$data_dir"/tst2012.fr.tok 

echo "Tokenizing test set (2013)"
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "en" < "$data_dir"/tst2013.en.norm > "$data_dir"/tst2013.en.tok 
echo "..."
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "fr" < "$data_dir"/tst2013.fr.norm> "$data_dir"/tst2013.fr.tok 

echo "Tokenizing test set (2014)"
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "en" < "$data_dir"/tst2014.en.norm > "$data_dir"/tst2014.en.tok 
echo "..."
perl "$code_dir"/tokenizer_apos.perl -threads 5 -l "fr" < "$data_dir"/tst2014.fr.norm> "$data_dir"/tst2014.fr.tok 