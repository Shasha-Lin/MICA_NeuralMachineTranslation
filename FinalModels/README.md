## Main model

For the main model, you can run ```train.py``` directly. This file was developed and teste on python3, and depends on the following packages: 

 - python3/intel/3.5.3
 - pytorch/python3.5/0.2.0_3
 - torchvision/python3.5/0.1.9

The code was tested and developed exclusivly using CUDA. 

It takes as parameters the following: 
```python 
parser.add_argument('--MIN_LENGTH', type=int, default=5, help='Min Length of sequence (Input side)')
parser.add_argument('--MAX_LENGTH', type=int, default=200, help='Max Length of sequence (Input side)')
parser.add_argument('--MIN_LENGTH_TARGET', type=int, default=5, help='Min Length of sequence (Output side)')
parser.add_argument('--MAX_LENGTH_TARGET', type=int, default=200, help='Max Length of sequence (Output side)')
parser.add_argument('--lang1', type=str, default="en", help='Input Language')
parser.add_argument('--lang2', type=str, default="fr", help='Target Language')
parser.add_argument('--USE_CUDA', action='store_true', help='IF USE CUDA (Default == False)')
parser.add_argument('--hidden_size', type=int, default=1024, help='Size of hidden layer')
parser.add_argument('--n_epochs', type=int, default=50000, help='Number of single iterations through the data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate (for both, encoder and decoder)')
parser.add_argument('--n_layers', type=int, default=2, help='Number of layers (for both, encoder and decoder)')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout (%) in the decoder')
parser.add_argument('--model_type', type=str, default="bpe2bpe", help='Model type (and ending of files)')
parser.add_argument('--main_data_dir', type=str, default= "/scratch/eff254/NLP/Data/Model_ready/", help='Directory where data is saved (in folders tain/dev/test)')
parser.add_argument('--out_dir', type=str, default="./checkpoints", help="Directory to save the models state dict (No default)")
parser.add_argument('--optimizer', type=str, default="Adam", help="Optimizer (Adam vs SGD). Default: Adam")
parser.add_argument('--kmax', type=int, default=10, help="Beam search Topk to search")
parser.add_argument('--clip', type=int, default=1, help="Clipping the gradients")
parser.add_argument('--batch_size', type=int, default=80, help="Size of a batch")
parser.add_argument('--attention', type=str, default='Bahdanau', help='attention type: either Bahdanau or Luong')
parser.add_argument('--scheduled_sampling_k', type=int, default=3000, help='scheduled sampling parameter for teacher forcing, based on inverse sigmoid decay')
parser.add_argument('--eval_dir', type=str, default="/scratch/eff254/NLP/Evaluation/", help="Directory to save predictions - MUST CONTAIN PEARL SCRIPT")
parser.add_argument('--experiment', type=str, default="MICA", help="Experiment name (for comet_ml purposes)")
```

The code has hard-coded checkpoints every 200 iterations on the data. This checkpoints save the encoder.state_dict(), decoder.state_dict(), encoder_optimizer.state_dict() and decoder_optimizer.state_dict() in the specified directory. This directory will automatically be created inside ```out_dir``` with the provided experiment name.This directoy will also received once the Lang classes for the Input and target languages, which contain the dictionaries ```word2index``` and ```index2word```.

This code also assummes comet.ml is being used. Currently it has hard-coded the API-Key to the team account. You can install comet.ml with ```python3 pip install comet```

## Run files

All the experiments were run in NYU's Prince nodes. If you are unfamiliar with the cluster, visit: https://wikis.nyu.edu/display/NYUHPC/Clusters+-+Prince. 

The cluster uses Slurm Workload Manager to submitt jobs. The file ```run_job.sh``` submits the train.py, requesting one GPU and 48 hours to train. This takes as parameters the model type ("bpe2bpe" or "bpe2char"), the learning rate, hidden size, kmax, number of layers, the code directory of the train.py file, the attention to use ("Luong" or "Bahdanau") and the experiment name. An example to run succesfully run a job is: 

```sh
sbatch run_job.sh "bpe2bpe" "0.00001" "512" "15" "2" "/scratch/USERID/train.py"
```
