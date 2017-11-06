#!/bin/bash

code_dir="/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/MICA_NeuralMachineTranslation/Preprocess/"

python "$code_dir"/file_merger.py --file1 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/train.en.tok.bpe" \
                                  --file2 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/train.fr.tok.bpe" \
                                  --outdir "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model1_ready/train/" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".bpe2bpe"
                                  
python "$code_dir"/file_merger.py --file1 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/dev.en.tok.bpe" \
                                  --file2 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/dev.fr.tok.bpe" \
                                  --outdir "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model1_ready/dev" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".bpe2bpe"
                                  
python "$code_dir"/file_merger.py --file1 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/valid.en.tok.bpe" \
                                  --file2 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/valid.fr.tok.bpe" \
                                  --outdir "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model1_ready/valid" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".bpe2bpe"
                                  
python "$code_dir"/file_merger.py --file1 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2010.en.tok.bpe" \
                                  --file2 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2010.fr.tok.bpe" \
                                  --outdir "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model1_ready/test" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".tst2010-bpe2bpe"
                                  
python "$code_dir"/file_merger.py --file1 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2011.en.tok.bpe" \
                                  --file2 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2011.fr.tok.bpe" \
                                  --outdir "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model1_ready/test" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".tst2011-bpe2bpe"
                                  
python "$code_dir"/file_merger.py --file1 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2012.en.tok.bpe" \
                                  --file2 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2012.fr.tok.bpe" \
                                  --outdir "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model1_ready/test" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".tst2012-bpe2bpe"
                                  
python "$code_dir"/file_merger.py --file1 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2013.en.tok.bpe" \
                                  --file2 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2013.fr.tok.bpe" \
                                  --outdir "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model1_ready/test" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".tst2013-bpe2bpe"
                                  
python "$code_dir"/file_merger.py --file1 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2014.en.tok.bpe" \
                                  --file2 "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/tst2014.fr.tok.bpe" \
                                  --outdir "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model1_ready/test" \
                                  --lang1 "en" \
                                  --lang2 "fr" \
                                  --term ".tst2014-bpe2bpe"
                                  