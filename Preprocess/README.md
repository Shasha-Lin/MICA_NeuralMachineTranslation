# Data pre-processing.

INTERNAL COMMENT: ALL FILES WERE UPLODED TO PRINCE; YOU NOW HAVE READ/WRITE/EXECUTE ACCESS. THE DIRECTORY IS: 
/scratch/eff254/NLP/Data/

CAUTION: NO SHUFFLE WAS MADE IN THE PROCESS; JUST RANDOMLY SPLITTED TRAIN/VALIDATION

This folder contains the codes to preprocess the Data.
All data was downloded form the following URL: 
  - https://wit3.fbk.eu/mt.php?release=2016-01 
  - Only the English to french Data was used

### General data cleaners. 
---
To clean the data, run the following scripts: 
```sh
$ python TrainCleaner.py
```
```sh
$ python ValidTestCleaner.py
```

On top of each scripts, the file params must be specified, indlucing train/validation split and directories to the original and output data. 

Both scripts depend on the following packages: 
``` python
import numpy as np
import re 
import pickle
```

```TrainCleaner.py```, using defualt parameters, output the following files: 
  - train_valid_metadata.p
  - train.en
  - train.fr
  - valid.en
  - valid.fr
 
In this case, valid is simply another random split on the provided train data. The metada is a list of lists, identified by order, of the form [speakers, keywords, urls, talkid], where each internal list have the same length (i.e. len(speakers) == len(keywords))

```ValidTestCleaner.py```, using defualt parameters, output the following files: 
  - tst2010.en
  - tst2010.en_metadata.p
  - tst2010.fr
  - tst2010.fr_metadata.p
  - tst2011.en
  - tst2011.en_metadata.p
  - tst2011.fr
  - tst2011.fr_metadata.p
  - tst2012.en
  - tst2012.en_metadata.p
  - tst2012.fr
  - tst2012.fr_metadata.p
  - tst2013.en
  - tst2013.en_metadata.p
  - tst2013.fr
  - tst2013.fr_metadata.p
  - tst2014.en
  - tst2014.en_metadata.p
  - tst2014.fr
  - tst2014.fr_metadata.p
  - dev.en
  - dev.en_metadata.p
  - dev.fr
  - dev.fr_metadata.p

This are the test and dev set provided directly from https://wit3.fbk.eu/mt.php?release=2016-01. The metada has the same form described above, a list of lists of the shape [speakers, keywords, urls, talkid]. 

### Normalization
---

Following the above codes, the ```normalize-punctuation.perl``` code is run. This code is from https://github.com/nyu-dl/dl4mt-cdec/tree/master/preprocess. 

To run it, just run the *runner* file, making sure to change the parameters and the beginning of the code: 
```sh
$ bash normalize-punctuation-runner.sh
```
This code outputs the files with ".norm" termination. 

### Tokenization
---

Having the normalization files, the ```tokenizer_apos.perl``` code is run. This code is, again, from https://github.com/nyu-dl/dl4mt-cdec/tree/master/preprocess. 

To run it, just run the *runner* file, again making sure to change the parameters and the beginning of the code: 
```sh
$ bash tokenizer_apos-runner.sh
```

This tokenizer calls the files in ```nonbreaking_prefixes```, which were also downloded from https://github.com/nyu-dl/dl4mt-cdec/tree/master/preprocess. 

### BPE
---

BPEs are learned from the train files and applied to valid, dev and test files. In this case, the main files ```apply_bpe.py``` and ```learn_bpe.py```are exact copies of https://github.com/rsennrich/subword-nmt. 

This files can be run with the following script: 
```sh
$ bash BPE-runner.sh
```
This outputs the files train.en.bpe and train.fr.bpe, which are the english and french BPE's learned from the data. After that, it applies them to the ".tok" files and outputs the files with termination ".tok.bpe". 

### Model Prep - Making Pairs
---

Finally, pairs are build and exporter using ```file_merger.py```. To run it, just run the *runner* file, modifying local directories inside this shell file. 
This files can be run with the following script: 
```sh
$ bash file_merge-runner-all.sh
```

This file requires ```argparse``` to be installed locally.

This code creates the pairs for a seq2seq model, using the tokenized files, the pairs for a bpe2bpe model using the tokenized and bpe files, and the pairs for a bpe2char model using the tokenized and bpe files. You will need to specify the location of the data directories within the file. 

