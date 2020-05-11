import argparse
import os

cur_path = os.path.dirname(__file__)

class config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--bert_config_file", default='/home/nianxw/nxw/chinese_L-12_H-768_A-12/bert_config.json', type=str,
                            help="The config json file corresponding to the pre-trained BERT model. "
                                 "This specifies the model architecture.")
        parser.add_argument("--vocab_file", default='/home/nianxw/nxw/chinese_L-12_H-768_A-12/vocab.txt', type=str,
                            help="The vocabulary file that the BERT model was trained on.")
        parser.add_argument("--init_checkpoint", default=None, type=str,
                            help="Initial checkpoint (usually from a pre-trained BERT model).")
     #    parser.add_argument("--init_checkpoint", default='/home/nianxw/nxw/cmrc_nxw/chinese_L-12_H-768_A-12/pytorch_model.bin', type=str,
     #                        help="Initial checkpoint (usually from a pre-trained BERT model).")

        ## Required parameters
        parser.add_argument("--output_dir", default=os.path.join(cur_path, 'output'), type=str,
                            help="The output directory where the model checkpoints and predictions will be written.")

        ## Other parameters
        parser.add_argument("--train_file", default=os.path.join(cur_path, 'data/cmrc2019_train.json'), type=str,
                            help="SQuAD json for training. E.g., train-v1.1.json")
        parser.add_argument("--predict_file", default=os.path.join(cur_path, 'data/cmrc2019_trial.json'), type=str,
                            help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
        parser.add_argument("--max_seq_length", default=512, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                 "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=64, type=int,
                            help="When splitting up a long document into chunks, how much stride to take between chunks.")
        parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
        parser.add_argument("--do_predict", default=False, action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
        parser.add_argument("--predict_batch_size", default=32, type=int, help="Total batch size for predictions.")
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs", default=3.0, type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion", default=0.1, type=float,
                            help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                                 "of training.")
        parser.add_argument("--verbose_logging", default=False, action='store_true',
                            help="If true, all of the warnings related to data processing will be printed. "
                                 "A number of warnings are expected for a normal SQuAD evaluation.")
        parser.add_argument("--no_cuda",
                            default=False,
                            help="Whether not to use CUDA when available")
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--do_lower_case",
                            default=True,
                            action='store_true',
                            help="Whether to lower case the input text. True for uncased models, False for cased models.")
        parser.add_argument("--local_rank",
                            type=int,
                            default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument('--fp16',
                            default=False,
                            action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--loss_scale',
                            type=float, default=0,
                            help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                 "0 (default value): dynamic loss scaling.\n"
                                 "Positive power of 2: static loss scaling value.\n")
        self.parser = parser

