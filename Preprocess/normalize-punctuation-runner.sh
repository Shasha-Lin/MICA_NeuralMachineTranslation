#!/bin/bash

#### CAREFEULL WITH DOUBLE QUOTATION IN SPACED DIRECTORY!!!
code_dir="/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Code_Own"
data_dir="/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess"


#### Normalize Punctuation Runner.

echo "Normalizing punctuation on validation set"
perl "$code_dir"/normalize-punctuation.perl -l "en" < "$data_dir"/valid.en > "$data_dir"/valid.en.norm
perl "$code_dir"/normalize-punctuation.perl -l "fr" < "$data_dir"/valid.fr > "$data_dir"/valid.fr.norm

echo "Normalizing punctuation on train set"
perl "$code_dir"/normalize-punctuation.perl -l "en" < "$data_dir"/train.en > "$data_dir"/train.en.norm
perl "$code_dir"/normalize-punctuation.perl -l "fr" < "$data_dir"/train.fr > "$data_dir"/train.fr.norm

echo "Normalizing punctuation on dev set"
perl "$code_dir"/normalize-punctuation.perl -l "en" < "$data_dir"/dev.en > "$data_dir"/dev.en.norm
perl "$code_dir"/normalize-punctuation.perl -l "fr" < "$data_dir"/dev.fr > "$data_dir"/dev.fr.norm

echo "Normalizing punctuation on tests sets"
perl "$code_dir"/normalize-punctuation.perl -l "en" < "$data_dir"/tst2010.en > "$data_dir"/tst2010.en.norm
perl "$code_dir"/normalize-punctuation.perl -l "fr" < "$data_dir"/tst2010.fr > "$data_dir"/tst2010.fr.norm

perl "$code_dir"/normalize-punctuation.perl -l "en" < "$data_dir"/tst2011.en > "$data_dir"/tst2011.en.norm
perl "$code_dir"/normalize-punctuation.perl -l "fr" < "$data_dir"/tst2011.fr > "$data_dir"/tst2011.fr.norm

perl "$code_dir"/normalize-punctuation.perl -l "en" < "$data_dir"/tst2012.en > "$data_dir"/tst2012.en.norm
perl "$code_dir"/normalize-punctuation.perl -l "fr" < "$data_dir"/tst2012.fr > "$data_dir"/tst2012.fr.norm

perl "$code_dir"/normalize-punctuation.perl -l "en" < "$data_dir"/tst2013.en > "$data_dir"/tst2013.en.norm
perl "$code_dir"/normalize-punctuation.perl -l "fr" < "$data_dir"/tst2013.fr > "$data_dir"/tst2013.fr.norm

perl "$code_dir"/normalize-punctuation.perl -l "en" < "$data_dir"/tst2014.en > "$data_dir"/tst2014.en.norm
perl "$code_dir"/normalize-punctuation.perl -l "fr" < "$data_dir"/tst2014.fr > "$data_dir"/tst2014.fr.norm
