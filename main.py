import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange
import re

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.nn import NLLLoss
from torch.nn import functional as F

from Config import config
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import PreTrainedBertModel,BertModel,BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from data import read_examples, convert_examples_to_features, write_predictions

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForMultiChoice(PreTrainedBertModel):

    def __init__(self, config):
        super(BertForMultiChoice, self).__init__(config)
        self.bert = BertModel(config)
        self.W1 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.W2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.apply(self.init_bert_weights)

    def gather_info(self, input_tensor, positions):
        batch_size, seq_len, hidden_size = input_tensor.size()
        flat_offsets = torch.linspace(0, batch_size-1, steps=batch_size).long().view(-1, 1)*seq_len
        flat_positions = positions.long() + flat_offsets
        flat_positions = flat_positions.view(-1)
        flat_seq_tensor = input_tensor.view(batch_size*seq_len, hidden_size)
        output_tensor = torch.index_select(flat_seq_tensor, 0, flat_positions).view(batch_size, -1, hidden_size)
        return output_tensor

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                choice_positions=None,  # (batch_size, choice_nums) [[1,2,5,7,8...], ...]
                answer_poisitions=None,  # (batch_size, answer_nums)
                labels=None,
                labels_mask=None,
                limit_loss=False):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        # get choice representation
        choice_rep = self.gather_info(sequence_output, choice_positions)
        choice_rep = choice_rep.split(2, 1)
        choice_rep = list(map(lambda x: x.view(x.size()[0], -1), choice_rep))
        choice_rep = torch.stack(choice_rep, 1)
        choice_rep_w1 = self.W1(choice_rep)   # [batch_size, m, hidden_size]  m为候选答案数目

        # get answer representation
        answer_rep = self.gather_info(sequence_output, answer_poisitions)
        answer_rep_w2 = self.W2(answer_rep)   # [batch_size, n, hidden_size]  n为blank数目

        attention_matrix = torch.bmm(choice_rep_w1, answer_rep_w2.permute(0, 2, 1))
        logits = F.softmax(attention_matrix, -1)

        if labels is not None:
            # get loss
            # loss1
            logits_flat = torch.log(logits.view(-1, logits.size()[-1]))
            labels = labels.view(-1)
            loss_fct = NLLLoss()
            loss1 = loss_fct(logits_flat, labels)
            total_loss = loss1
            
            # loss2
            if limit_loss:
                logits_normal = F.normalize(logits, p=2, dim=-1)
                choice_att_matrix = torch.bmm(logits_normal, logits_normal.permute(0, 2, 1))   # [batch_size, m, m]
                # choice_att_matrix = torch.triu(choice_att_matrix, 1).view(choice_att_matrix.size()[0], -1)
                choice_att_matrix = torch.triu(choice_att_matrix, 1)

                # 获取两个子loss
                labels_mask_1 = labels_mask.permute(0, 2, 1)
                cos_distance_1 = choice_att_matrix.mul(labels_mask_1).view(choice_att_matrix.size()[0], -1)

                labels_mask_2 = 1 - (labels_mask_1 + labels_mask)
                cos_distance_2 = 1 - choice_att_matrix.mul(labels_mask_2).view(choice_att_matrix.size()[0], -1)

                cos_distance = torch.sum(cos_distance_1, dim=-1).squeeze() + torch.sum(cos_distance_2, dim=-1).squeeze()
                loss2 = cos_distance.mean()
                total_loss += loss2
            return loss1, loss2
        else:
            return logits

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "logits"])

def main():
    args = config().parser.parse_args()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
        raw_train_data = json.load(open(args.train_file, mode='r'))
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")
        raw_test_data = json.load(open(args.predict_file, mode='r'))

    if os.path.exists(args.output_dir)==False:
        # raise ValueError("Output directory () already exists and is not empty.")
        os.makedirs(args.output_dir, exist_ok=True)
    
    import pickle as cPickle
    train_examples = None
    num_train_steps = None
    if args.do_train:
        if os.path.exists("train_file_baseline.pkl"):
            train_examples=cPickle.load(open("train_file_baseline.pkl",mode='rb'))
        else:
            train_examples = read_examples(raw_train_data, doc_stride=args.doc_stride, max_seq_length=args.max_seq_length, is_training=True)
            cPickle.dump(train_examples,open("train_file_baseline.pkl",mode='wb'))
        logger.info("train examples {}".format(len(train_examples)))
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)


    # Prepare model
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    model =BertForMultiChoice(bert_config)
    if args.init_checkpoint is not None:
        logger.info('load bert weight')
        state_dict=torch.load(args.init_checkpoint, map_location='cpu')
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        # new_state_dict=state_dict.copy()
        # for kye ,value in state_dict.items():
        #     new_state_dict[kye.replace("bert","c_bert")]=value
        # state_dict=new_state_dict
        if metadata is not None:
            state_dict._metadata = metadata
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                # logger.info("name {} chile {}".format(name,child))
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        logger.info("missing keys:{}".format(missing_keys))
        logger.info('unexpected keys:{}'.format(unexpected_keys))
        logger.info('error msgs:{}'.format(error_msgs))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)


    global_step = 0
    if args.do_train:
        cached_train_features_file = args.train_file+'_{0}_{1}_v{2}'.format(str(args.max_seq_length), str(args.doc_stride), str(1))
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                is_training=True)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_choice_positions = torch.tensor([f.choice_positions for f in  train_features],dtype=torch.long)
        all_answer_positions = torch.tensor([f.answer_positions for f in  train_features],dtype=torch.long)
        all_choice_labels = torch.tensor([f.choice_labels for f in train_features], dtype=torch.long)
        all_choice_labels_mask = torch.tensor([f.all_choice_labels_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, 
                                   all_segment_ids,all_choice_positions,
                                   all_answer_positions, all_choice_labels, all_choice_labels_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.zero_grad()
            epoch_itorator=tqdm(train_dataloader,disable=None)
            for step, batch in enumerate(epoch_itorator):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, all_choice_positions, all_answer_positions, all_choice_labels = batch
                loss1, loss2 = model(input_ids, segment_ids, input_mask, all_choice_positions, all_answer_positions, all_choice_labels, all_choice_labels_mask, limit_loss=True)
                loss = loss1 + loss2
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (step+1) % 50 == 0:
                    logger.info("step: {} #### loss1: {}  loss2: {}".format(step,loss1.cpu().item(), loss2.cpu().item()))

    # Save a trained model
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)


    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model =BertForMultiChoice(bert_config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_examples(
            raw_test_data, doc_stride=args.doc_stride, max_seq_length=args.max_seq_length, is_training=False)
        # eval_examples=eval_examples[:100]
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            is_training=False)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_choice_positions = torch.tensor([f.choice_positions for f in  eval_features],dtype=torch.long)
        all_answer_positions = torch.tensor([f.answer_positions for f in  eval_features],dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, 
                                  all_segment_ids,all_choice_positions,
                                  all_answer_positions, all_example_index)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
        
        
        model.eval()
        all_results = []
        logger.info("Start evaluating")

        for input_ids, input_mask, segment_ids, choice_positions, answer_positions, example_indices in tqdm(eval_dataloader, desc="Evaluating",disable=None):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            choice_positions = choice_positions.to(device)
            answer_positions = answer_positions.to(device)
            with torch.no_grad():
                batch_logits = model(input_ids, input_mask, segment_ids, choice_positions, answer_positions)  # [24, n]
            for i, example_index in enumerate(example_indices):
                logits = batch_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             logits=logits))
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        
        
        write_predictions(eval_examples, eval_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, args.verbose_logging)


if __name__ == "__main__":
    main()