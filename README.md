##  MICA_NeuralMachineTranslation
Team MICA's attempt at NMT

Dataset: English to French: https://wit3.fbk.eu/mt.php?release=2016-01

## Todo: 
- Data Processing https://wit3.fbk.eu/mt.php?release=2016-01 : Eduardo
  - Both models: BPE on source
  - Model 1 Target: Use BPE
  - Model 2 Target: Does not use BPE

- Model 1: Encoder-Decoder With Attention & BPE 
  - Adapt from Theano to pytorch : Akash

- Model 2: Character-level Decoder 
  - Adapt from Theano to pytorch : Millie

- Next meeting: Thursday 

## References:
- Original BPE implementation: https://github.com/rsennrich/subword-nmt
- Character level: https://github.com/nyu-dl/dl4mt-c2c
- General Tutorial: https://github.com/nyu-dl/dl4mt-tutorial
- This tutorial is supposed to be helpful (from  NLP reading list): https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/
- Pytorch implementation of attention translation models https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb

---
## Comet ML (account for everyone): 
- email: eff254@nyu.edu 
- user: edufierro
- password: wearemica
- API Key: 00Z9vIf4wOLZ0yrqzdwHqttv4
