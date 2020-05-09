import json
import os
import re
import random
import logging
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# 训练时，所有候选答案都应该用上，这样才能与预测相对应，doc句子长度比较短
# 先使用BERT词表


cur_path = os.path.dirname(os.path.realpath(__file__))

class SenExample(object):
    """A single training/test example for the Squad dataset."""

    def __init__(self,
                 qas_id,
                 example_index,
                 unique_id,
                 choices,
                 sub_doc_texts,
                 sub_answer_texts,
                 choice_labels=None,
                 choice_labels_for_consine=None):
        self.qas_id = qas_id
        self.example_index = example_index
        self.unique_id = unique_id
        self.choices = choices
        self.sub_doc_texts = sub_doc_texts
        self.sub_answer_texts = sub_answer_texts
        self.choice_labels = choice_labels
        self.choice_labels_for_consine = choice_labels_for_consine

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", choices: %s" % ( " ".join(
            self.choices))
        s += ", sub_doc_texts: [%s]" % (" ".join(self.sub_doc_texts))
        if self.choice_labels:
            s += ", choice_labels: %d" % (self.choice_labels)
        return s



def read_examples(input_data, doc_stride=64, max_seq_length=512, is_training=True):
    replace_list=[
       ['[BLANK1]', '[unused1]'],
       ['[BLANK2]', '[unused2]'],
       ['[BLANK3]', '[unused3]'],
       ['[BLANK4]', '[unused4]'],
       ['[BLANK5]', '[unused5]'],
       ['[BLANK6]', '[unused6]'],
       ['[BLANK7]', '[unused7]'],
       ['[BLANK8]', '[unused8]'],
       ['[BLANK9]', '[unused9]'],
       ['[BLANK10]', '[unused10]'],
       ['[BLANK11]', '[unused11]'],
       ['[BLANK12]', '[unused12]'],
       ['[BLANK13]', '[unused13]'],
       ['[BLANK14]', '[unused14]'],
       ['[BLANK15]', '[unused15]'],
       ]

    examples = []
    unique_id = 100000000000
    example_index = 0
    for entry in tqdm(input_data['data'][:1000]):
        # if examples_count % 1000 == 0:
        #     print("已生成 %d 条样本" % examples_count)
        paragraph = entry
        context_index = entry["context_id"]

        # 处理文章数据
        paragraph_text = paragraph["context"]
        for key, value in replace_list:
            paragraph_text = paragraph_text.replace(key,value,1)
        tmp_text = ""
        is_blank = False
        for word_index in range(len(paragraph_text)):
            word = paragraph_text[word_index]
            if paragraph_text[word_index:word_index+7]=="[unused":
                is_blank = True
            if word.strip() == ']':
                is_blank = False
            if is_blank is False:
                tmp_text += word.strip() + " "
            else:
                tmp_text += word.strip()
        # 把句子中的字按空格划分，其中[BLANK8]以及被替换为[unused8]，且作为一个整体
        doc_texts = tmp_text.strip().split()

        # 构建choice-answer字典
        choice_to_answer = {}
        answer_to_choice = {}
        answers = entry["answers"]
        choices = entry["choices"]
        for answer_index, choice_num in enumerate(answers):
            # blank对应文本，用于确定位置
            answer_text = "[unused{}]".format(str((answer_index)+1))
            choice_to_answer[choice_num] = answer_text
            answer_to_choice[answer_text] = choice_num
        
        choice_length = 0
        choice_nums = len(choices)
        for choice in choices:
            choice_length += len(choice)
        # 制作样本，滑动窗口设置为doc_stride
        start_index = 0
        span_length = max_seq_length - (choice_length + 2*choice_nums + 3)
        while start_index < len(doc_texts):           
            end_index = start_index + span_length
            if end_index > len(doc_texts):
                end_index = len(doc_texts)
            sub_doc_texts = doc_texts[start_index: end_index]

            sub_answers = []
            for item in sub_doc_texts:
                if "[unused" in item:
                    sub_answers.append(item)
            choice_labels = []
            choice_labels_for_consine = []
            if is_training:
                for i in range(len(choices)):
                    a = choice_to_answer[i]
                    if a in sub_answers:
                        choice_labels.append(sub_answers.index(a)+1)
                        choice_labels_for_consine.append(0)
                    else:
                        choice_labels.append(0)
                        choice_labels_for_consine.append(1)
            example = SenExample(
                qas_id=str(context_index)+"###"+str(start_index)+"#"+str(end_index),
                example_index=example_index,
                unique_id=unique_id,
                choices=choices,
                sub_doc_texts=sub_doc_texts,
                sub_answer_texts=sub_answers,
                choice_labels=choice_labels,
                choice_labels_for_consine=choice_labels_for_consine
            )
            examples.append(example)
            unique_id += 1
            start_index += doc_stride
        example_index += 1
    return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 choice_positions,
                 answer_positions,
                 choice_positions_mask,
                 answer_positions_mask,
                 choice_labels=None,
                 choice_labels_for_consine=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.choice_positions = choice_positions
        self.answer_positions = answer_positions
        self.choice_positions_mask = choice_positions_mask
        self.answer_positions_mask = answer_positions_mask
        self.choice_labels = choice_labels
        self.choice_labels_mask = choice_positions_mask
        self.choice_labels_for_consine = choice_labels_for_consine


def convert_examples_to_features(examples, tokenizer, max_choice_nums=20, 
                                 max_answer_nums=15, max_seq_length=512, is_training=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in tqdm(examples):
        unique_id = example.unique_id
        example_index = example.example_index
        doc_span_index = example.qas_id
        doc_tokens = example.sub_doc_texts  # 文本
        doc_choice_texts = [tokenizer.tokenize(_) for _ in example.choices]  # 候选答案
        doc_choice_labels = example.choice_labels
        doc_choice_labels_for_consine = example.choice_labels_for_consine

        tokens = []  # 输入序列
        choice_positions = []  # choice的在input中对应的起始和终止位置
        answer_positions = []  # answer在input中对应的位置
        segment_ids = []  # seg id

        tokens.append("[CLS]")
        for i, choice in enumerate(doc_choice_texts):
            choice_positions.append(len(tokens))
            tokens.append("[unused"+str(50+i+1)+"]")  # BERT词表找对应标识,从[unused51]开始
            for word in choice:
                tokens.append(word)

        tokens.append("[SEP]")
        for _ in range(len(tokens)):
            segment_ids.append(0)

        for word in doc_tokens:
            if "[unused" in word:
                answer_positions.append(len(tokens))
            tokens.append(word)
            segment_ids.append(1)

        tokens.append('[SEP]')
        segment_ids.append(1)


        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        choice_positions_mask = [1] * len(choice_positions)
        while len(choice_positions) < max_choice_nums:
            choice_positions.append(0)
            doc_choice_labels.append(0)
            doc_choice_labels_for_consine.append(0)
            choice_positions_mask.append(0)

        answer_positions_mask = [1] * len(answer_positions)
        while len(answer_positions) < max_answer_nums:
            answer_positions.append(max_seq_length-1)
            answer_positions_mask.append(0)
        
        assert len(choice_positions) == max_choice_nums
        assert len(choice_positions_mask) == max_choice_nums
        assert len(answer_positions) == max_answer_nums
        assert len(answer_positions_mask) == max_answer_nums
        assert len(doc_choice_labels) == max_answer_nums
        assert len(doc_choice_labels_for_consine) == max_answer_nums



        if example_index < 3:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (unique_id))
            logger.info("example_index: %s" % (example_index))
            logger.info("doc_span_index: %s" % (doc_span_index))
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("choice_positions: %s" % " ".join([str(x) for x in choice_positions]))
            logger.info("answer_positions: %s" % " ".join([str(x) for x in answer_positions]))
            logger.info(
                "choice_positions_mask: %s" % " ".join([str(x) for x in choice_positions_mask]))
            logger.info(
                "choice_positions_mask: %s" % " ".join([str(x) for x in answer_positions_mask]))

            if is_training:
                logger.info("choice_labels: %s" % " ".join([str(x) for x in doc_choice_labels]))
                logger.info("choice_labels: %s" % " ".join([str(x) for x in doc_choice_labels_for_consine]))

        features.append(InputFeatures(
            unique_id=unique_id,
            example_index=example_index,
            doc_span_index=doc_span_index,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            choice_positions=choice_positions,
            answer_positions=answer_positions,
            answer_positions_mask=answer_positions_mask,
            choice_positions_mask=choice_positions_mask,
            choice_labels=doc_choice_labels,
            choice_labels_for_consine=doc_choice_labels_for_consine
        )) 


    return features

def write_predictions():
    return