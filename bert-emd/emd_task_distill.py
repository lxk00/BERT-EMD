# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""
from __future__ import absolute_import, division, print_function
import time

import argparse
import csv
import logging
import os
import random
import sys
import profile
import shutil
from pyemd import emd_with_flow
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from bert_fineturn.data_processor.glue import glue_compute_metrics as compute_metrics
from bert_fineturn.data_processor.glue import glue_output_modes as output_modes
from bert_fineturn.data_processor.glue import glue_processors as processors
from tinybert.modeling import TinyBertForSequenceClassification, BertConfig, TinyBertForPreTraining
from tinybert.tokenization import BertTokenizer
from tinybert.optimization import BertAdam
from tinybert.file_utils import WEIGHTS_NAME, CONFIG_NAME

time.sleep(random.random())
csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def do_eval(args, model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    # for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
    for batch_ in eval_dataloader:
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _, _ = model(input_ids, segment_ids, input_mask, is_conv=args.is_conv, share_param=args.share_param)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss

    return result

def do_predict(args, model, device, output_mode, tokenizer):
    task_name = args.task_name.lower()
    pred_task_names = ("mnli", "mnli-mm") if task_name == "mnli" else (task_name,)
    pred_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if task_name == "mnli" else (args.output_dir,)
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        if not os.path.exists(pred_output_dir):
            os.mkdir(pred_output_dir)
        processor = processors[pred_task]()
        label_list = processor.get_labels()
        pred_examples = processor.get_test_examples(args.data_dir)
        pred_features = convert_examples_to_features(pred_examples, label_list, args.max_seq_length, tokenizer,
                                                     output_mode)
        pred_data, pred_labels = get_tensor_data(output_mode, pred_features)
        pred_sampler = SequentialSampler(pred_data)
        pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=args.eval_batch_size)
        logger.info("  Num examples = %d", len(pred_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        preds = []
        for batch_ in tqdm(pred_dataloader, desc="predicting"):
            batch_ = tuple(t.to(device) for t in batch_)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
                logits, _, _ = model(input_ids, segment_ids, input_mask)

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

        label_map = {i: label for i, label in enumerate(label_list)}
        output_pred_file = os.path.join(pred_output_dir, pred_task.upper() + ".tsv")
        with open(output_pred_file, "w") as writer:
            logger.info("***** predict results *****")
            writer.write("index\tprediction\n")
            for index, pred in enumerate(tqdm(preds)):
                if pred_task == 'sts-b':
                    pred = round(pred, 3)
                else:
                    pred = label_map[pred]
                writer.write("%s\t%s\n" % (index, str(pred)))
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def get_new_layer_weight(trans_matrix, distance_matrix, stu_layer_num, tea_layer_num, T, type_update='att'):
    if type_update == 'att':
        global att_student_weight, att_teacher_weight
        student_layer_weight = np.copy(att_student_weight)
        teacher_layer_weight = np.copy(att_teacher_weight)
    else:
        global rep_student_weight, rep_teacher_weight
        student_layer_weight = np.copy(rep_student_weight)
        teacher_layer_weight = np.copy(rep_teacher_weight)

    distance_matrix = distance_matrix.detach().cpu().numpy().astype('float64')
    trans_weight = np.sum(trans_matrix * distance_matrix, -1)
    # logger.info('student_trans_weight:{}'.format(trans_weight))
    # new_student_weight = torch.zeros(stu_layer_num).cuda()
    for i in range(stu_layer_num):
        student_layer_weight[i] = trans_weight[i] / student_layer_weight[i]
    weight_sum = np.sum(student_layer_weight)
    for i in range(stu_layer_num):
        if student_layer_weight[i] != 0:
            student_layer_weight[i] = weight_sum / student_layer_weight[i]

    trans_weight = np.sum(np.transpose(trans_matrix) * distance_matrix, -1)
    for j in range(tea_layer_num):
        teacher_layer_weight[j] = trans_weight[j + stu_layer_num] / teacher_layer_weight[j]
    weight_sum = np.sum(teacher_layer_weight)
    for i in range(tea_layer_num):
        if teacher_layer_weight[i] != 0:
            teacher_layer_weight[i] = weight_sum / teacher_layer_weight[i]

    student_layer_weight = student_layer_weight / np.sum(student_layer_weight)
    teacher_layer_weight = teacher_layer_weight / np.sum(teacher_layer_weight)

    if type_update == 'att':
        att_student_weight = student_layer_weight
        att_teacher_weight = teacher_layer_weight
    else:
        rep_student_weight = student_layer_weight
        rep_teacher_weight = teacher_layer_weight


def transformer_loss(student_atts, teacher_atts, student_reps, teacher_reps,
                     device, loss_mse, args, global_step, T=1):
    global att_student_weight, rep_student_weight, att_teacher_weight, rep_teacher_weight
    def embedding_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j]
                tmp_loss = loss_mse(student_rep, teacher_rep)
                # tmp_loss = torch.nn.functional.normalize(tmp_loss, p=2, dim=2)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        # trans_matrix = trans_matrix
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix

    def emd_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i+1]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j + 1]
                tmp_loss = loss_mse(student_rep, teacher_rep)
                # tmp_loss = torch.nn.functional.normalize(tmp_loss, p=2, dim=2)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        # trans_matrix = trans_matrix
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix

    def emd_att_loss(student_atts, teacher_atts, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):

        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_att = student_atts[i]
            for j in range(tea_layer_num):
                teacher_att = teacher_atts[j]
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                          student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                          teacher_att)

                tmp_loss = loss_mse(student_att, teacher_att)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss
        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        att_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return att_loss, trans_matrix, distance_matrix

    stu_layer_num = len(student_atts)
    tea_layer_num = len(teacher_atts)
    if args.use_att:
        att_loss, att_trans_matrix, att_distance_matrix = \
            emd_att_loss(student_atts, teacher_atts, att_student_weight, att_teacher_weight,
                         stu_layer_num, tea_layer_num, device, loss_mse)
        if args.update_weight:
            get_new_layer_weight(att_trans_matrix, att_distance_matrix, stu_layer_num, tea_layer_num, T=T)
        att_loss = att_loss.to(device)
    else:
        att_loss = torch.tensor(0)
    if args.use_rep:
        if args.embedding_emd:
            rep_loss, rep_trans_matrix, rep_distance_matrix = \
                embedding_rep_loss(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                             stu_layer_num+1, tea_layer_num+1, device, loss_mse)
            if args.update_weight:
                get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num+1, tea_layer_num+1, T=T, type_update='xx')
        else:
            rep_loss, rep_trans_matrix, rep_distance_matrix = \
                emd_rep_loss(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                             stu_layer_num, tea_layer_num, device, loss_mse)

            if args.update_weight:
                get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num, T=T, type_update='xx')
        rep_loss = rep_loss.to(device)
    else:
        rep_loss = torch.tensor(0)


    if not args.seperate:
        student_weight = np.mean(np.stack([att_student_weight, rep_student_weight]), 0)
        teacher_weight = np.mean(np.stack([att_teacher_weight, rep_teacher_weight]), 0)
        if global_step % args.eval_step == 0:
            logger.info('all_student_weight:{}'.format(student_weight))
            logger.info('all_teacher_weight:{}'.format(teacher_weight))
        att_student_weight = student_weight
        att_teacher_weight = teacher_weight
        rep_student_weight = student_weight
        rep_teacher_weight = teacher_weight
    else:
        if global_step % args.eval_step == 0:
            logger.info('att_student_weight:{}'.format(att_student_weight))
            logger.info('att_teacher_weight:{}'.format(att_teacher_weight))
            logger.info('rep_student_weight:{}'.format(rep_student_weight))
            logger.info('rep_teacher_weight:{}'.format(rep_teacher_weight))

#       att_student_weight = att_student_weight / np.sum(att_student_weight)
#       att_teacher_weight = att_teacher_weight / np.sum(att_teacher_weight)

#       rep_student_weight = rep_student_weight / np.sum(rep_student_weight)
#       rep_teacher_weight = rep_teacher_weight / np.sum(rep_student_weight)
    return att_loss, rep_loss

def pkd_loss(student_atts, teacher_atts, student_reps, teacher_reps, device='cuda'):
    teacher_atts = [teacher_atts[i] for i in [1, 3, 5, 7, 9]]
    att_tmp_loss, rep_tmp_loss = [], []
    for student_att, teacher_att in zip(student_atts, teacher_atts):
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                    student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                    teacher_att)

        att_tmp_loss.append(torch.nn.functional.mse_loss(student_att, teacher_att))
    att_loss = sum(att_tmp_loss)
    new_teacher_reps = [teacher_reps[i] for i in [2, 4, 6, 8, 10]]
    new_student_reps = student_reps[1:-1]
    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
        rep_tmp_loss.append(torch.nn.functional.mse_loss(student_rep, teacher_rep))
    rep_loss = sum(rep_tmp_loss)

    return att_loss, rep_loss

def eval_milimm(args, device, global_step, label_list, num_labels, output_mode, student_model, tokenizer):

    task_name = "mnli-mm"
    processor = processors[task_name]()
    if not os.path.exists(args.output_dir + '-MM'):
        os.makedirs(args.output_dir + '-MM')
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    logger.info("***** Running mm evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)
    result = do_eval(args, student_model, task_name, eval_dataloader,
                     device, output_mode, eval_labels, num_labels)
    result['global_step'] = global_step
    tmp_output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
    result_to_file(result, tmp_output_eval_file)


def save_model(args, student_model, tokenizer, model_name):
    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
    output_model_file = os.path.join(args.output_dir, model_name)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)


def get_eval_result(args, device, eval_dataloader, eval_labels, global_step, num_labels, output_mode, step, student_model,
                    task_name, tr_att_loss, tr_cls_loss, tr_loss, tr_rep_loss):
    loss = tr_loss / (step + 1)
    cls_loss = tr_cls_loss / (step + 1)
    att_loss = tr_att_loss / (step + 1)
    rep_loss = tr_rep_loss / (step + 1)
    result = do_eval(args, student_model, task_name, eval_dataloader,
                     device, output_mode, eval_labels, num_labels)
    result['global_step'] = global_step
    result['cls_loss'] = cls_loss
    result['att_loss'] = att_loss
    result['rep_loss'] = rep_loss
    result['loss'] = loss
    return result

def distillation_loss(y, labels, teacher_scores, output_mode, T, alpha, reduction_kd='mean', reduction_nll='mean', reduce_T=1, is_teacher=True):
    teacher_T = T if is_teacher else 1
    rt = T*T/reduce_T if is_teacher else 1
    if output_mode == "classification":
        if teacher_scores is not None:
            student_likelihood = torch.nn.functional.log_softmax(y / T, dim=-1)
            targets_prob = torch.nn.functional.softmax(teacher_scores / T, dim=-1)
            d_loss = (- targets_prob * student_likelihood).mean() * T * T / reduce_T
        else:
            assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
            d_loss = 0.0
        nll_loss = torch.nn.functional.cross_entropy(y, labels, reduction=reduction_nll)
    elif output_mode == "regression":
        loss_mse = MSELoss()
        d_loss = loss_mse(y.view(-1), teacher_scores.view(-1))
        nll_loss = loss_mse(y.view(-1), labels.view(-1))
    else:
        assert output_mode in ["classification", "regression"]
        d_loss = 0.0
        nll_loss = 0.0
    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss
    return tol_loss, d_loss, nll_loss

def distillation_loss1(outputs, teacher_outputs, T):
    p_logit = outputs / T
    q_logit = teacher_outputs / T
    p = torch.nn.functional.softmax(p_logit, dim=-1)
    q = torch.nn.functional.softmax(q_logit, dim=-1)
    kd_loss = torch.sum(-q * torch.nn.functional.log_softmax(p_logit, dim=-1), 1)
    # kd_loss=torch.nn.KLDivLoss()(F.log_softmax(outputs/T,dim=2),F.softmax(teacher_outputs/T,dim=2))


def do_train(args):
    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if not args.do_predict and not args.do_eval and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        shutil.rmtree(args.output_dir, True)
        logger.info("exist Output directory ({}) removed.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    fw_args = open(args.output_dir + '/args.txt', 'w')
    fw_args.write(str(args))
    fw_args.close()

    task_name = args.task_name.lower()


    logger.info('\npred_distill:{}\nuse_emd:{}\nseperate:{}\ntrain_epoch:{}\n'.format(
        args.pred_distill, args.use_emd, args.seperate, args.num_train_epochs))

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if not args.do_eval:
        if not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir)
        else:
            train_examples = processor.get_aug_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)
        train_data, _ = get_tensor_data(output_mode, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if not args.do_eval:
        teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
        teacher_model.to(device)
    if not args.pkd:
        if args.no_pretrain:
            student_model = TinyBertForSequenceClassification.from_scratch(args.student_model, num_labels=num_labels)
        else:
            student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model, num_labels=num_labels)
    else:
        student_model = TinyBertForSequenceClassification.from_scratch(args.teacher_model, num_labels=num_labels, small_model=True)
    student_model.to(device)
    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(args, student_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    elif args.do_predict:
        logger.info("***** Running prediction *****")
        student_model.eval()
        do_predict(args, student_model, device, output_mode, tokenizer)
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        if not args.pred_distill:
            schedule = 'none'
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        loss_mse = MSELoss()
        def soft_cross_entropy(predicts, targets, T):
            student_likelihood = torch.nn.functional.log_softmax(predicts/T, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets/T, dim=-1)
            return (- targets_prob * student_likelihood).mean() * T ** 2 / 2

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        if task_name == 'cola':
            best_dev_acc = [-1, 0]

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        config = BertConfig.from_json_file(os.path.join(args.student_model, CONFIG_NAME))
        if args.pkd:
            config.num_hidden_layers = 6
        global att_student_weight, rep_student_weight, att_teacher_weight, rep_teacher_weight
        
        att_student_weight = np.ones(config.num_hidden_layers) / config.num_hidden_layers
        rep_student_weight = np.ones(config.num_hidden_layers) / config.num_hidden_layers
        config = BertConfig.from_json_file(os.path.join(args.teacher_model, CONFIG_NAME))
        att_teacher_weight = np.ones(config.num_hidden_layers) / config.num_hidden_layers
        rep_teacher_weight = np.ones(config.num_hidden_layers) / config.num_hidden_layers

        for epoch_ in range(int(args.num_train_epochs)):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True, miniters= int(len(train_dataloader)/4))):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                cls_loss = 0.
                student_logits, student_atts, student_reps = \
                    student_model(input_ids, segment_ids, input_mask, is_student=True,
                                  is_conv=args.is_conv, share_param=args.share_param)

                with torch.no_grad():
                    teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

                if args.one_step:
                    if args.use_emd:
                        att_loss, rep_loss = \
                            transformer_loss(student_atts, teacher_atts, student_reps, teacher_reps,
                                            device, loss_mse, args, global_step, T=args.T_emd)
                        embedding_loss = loss_mse(student_reps[0], teacher_reps[0])
                    elif args.tb_onestep:
                        att_loss = 0.
                        rep_loss = 0.
                        teacher_layer_num = len(teacher_atts)
                        student_layer_num = len(student_atts)
                        assert teacher_layer_num % student_layer_num == 0
                        layers_per_block = int(teacher_layer_num / student_layer_num)
                        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                            for i in range(student_layer_num)]

                        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                      student_att)
                            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                      teacher_att)

                            tmp_loss = loss_mse(student_att, teacher_att)
                            att_loss += tmp_loss
                        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                        new_student_reps = student_reps
                        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                            tmp_loss = loss_mse(student_rep, teacher_rep)
                            rep_loss += tmp_loss
                        embedding_loss = 0
                        # loss = rep_loss + att_loss

                    elif args.pkd:
                        att_loss, rep_loss = pkd_loss(student_atts, teacher_atts, student_reps, teacher_reps)
                        embedding_loss = loss_mse(student_reps[0], teacher_reps[0])
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()

                    if args.new_pred_loss:
                        cls_loss, kd_loss, ce_loss = distillation_loss(student_logits, label_ids, teacher_logits, output_mode, T=args.T, alpha=args.alpha, reduce_T=args.reduce_T, is_teacher=args.is_teacher)
                    else:
                        if output_mode == "classification":
                            cls_loss = soft_cross_entropy(student_logits, teacher_logits, args.T)
                        elif output_mode == "regression":
                            loss_mse = MSELoss()
                            cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                    tr_cls_loss += cls_loss.item()
                    if args.use_embedding and args.use_att and args.use_rep:
                        loss = args.beta * (args.theta * (rep_loss + att_loss) + embedding_loss) + cls_loss
                    elif args.use_att and args.use_rep:
                        loss = args.beta * (args.theta * (rep_loss + att_loss)) + cls_loss
                    elif args.use_embedding and args.use_att:
                        loss = args.beta * (args.theta * att_loss + embedding_loss) + cls_loss
                    elif args.use_embedding and args.use_rep:
                        loss = args.beta * (args.theta * rep_loss + embedding_loss) + cls_loss
                elif not args.pred_distill:
                    if args.use_emd:
                        att_loss, rep_loss, att_student_weight, att_teacher_weight, rep_student_weight, rep_teacher_weight = \
                            transformer_loss(student_atts, teacher_atts, student_reps, teacher_reps,
                                             device, loss_mse, args, global_step, args.T_emd)
                        embedding_loss = loss_mse(student_reps[0], teacher_reps[0])
                        loss = rep_loss + att_loss + embedding_loss
                        
                        tr_att_loss += att_loss.item()
                        tr_rep_loss += rep_loss.item()
                    else:
                        att_loss = 0.
                        rep_loss = 0.
                        teacher_layer_num = len(teacher_atts)
                        student_layer_num = len(student_atts)
                        assert teacher_layer_num % student_layer_num == 0
                        layers_per_block = int(teacher_layer_num / student_layer_num)
                        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                            for i in range(student_layer_num)]

                        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                      student_att)
                            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                      teacher_att)

                            tmp_loss = loss_mse(student_att, teacher_att)
                            att_loss += tmp_loss
                        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                        new_student_reps = student_reps
                        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                            tmp_loss = loss_mse(student_rep, teacher_rep)
                            rep_loss += tmp_loss

                        loss = rep_loss + att_loss
                        tr_att_loss += att_loss.item()
                        tr_rep_loss += rep_loss.item()
                else:
                    if args.new_pred_loss:
                        cls_loss, kd_loss, ce_loss = distillation_loss(student_logits, label_ids, teacher_logits, output_mode,
                                                                       T=args.T, alpha=args.alpha, reduce_T=args.reduce_T, is_teacher=args.is_teacher)
                    else:
                        if output_mode == "classification":
                            cls_loss = soft_cross_entropy(student_logits, teacher_logits, args.T)
                        elif output_mode == "regression":
                            loss_mse = MSELoss()
                            cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % args.eval_step == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    result = get_eval_result(args, device, eval_dataloader, eval_labels, global_step, num_labels, output_mode,
                                             step, student_model, task_name, tr_att_loss, tr_cls_loss, tr_loss,
                                             tr_rep_loss)
                    save_model(args, student_model, tokenizer, model_name='final_' + WEIGHTS_NAME)
                    result_to_file(result, output_eval_file)
                    if task_name == 'cola':
                        is_best = [False, False]
                    else:
                        is_best = False
                    
                    if task_name == 'sts-b':
                        if result['corr'] > best_dev_acc:
                            best_dev_acc = result['corr']
                            is_best = True

                    elif task_name == 'cola':
                        if result['mcc'] > best_dev_acc[0]:
                            best_dev_acc[0] = result['mcc']
                            is_best[0] = True
                        # if result['acc'] > best_dev_acc[1]:
                        #     best_dev_acc[1] = result['acc']
                        #     is_best[1] = True

                    elif result['acc'] > best_dev_acc:
                        best_dev_acc = result['acc']
                        is_best = True


                    if is_best and task_name != 'cola':
                        logger.info("***** Save model *****")
                        save_model(args, student_model, tokenizer, model_name=WEIGHTS_NAME)
                        result['best_acc'] = best_dev_acc
                        result_to_file(result, output_eval_file)

                        # Test mnli-mm
                        if args.pred_distill and task_name == "mnli":
                            eval_milimm(args, device, global_step, label_list, num_labels, output_mode, student_model, tokenizer)
                    else:
                        if type(is_best) is list:
                            if is_best[0]:
                                save_model(args, student_model, tokenizer, model_name='pytorch_model.bin')
                                result['best_mcc'] = best_dev_acc[0]
                                result_to_file(result, output_eval_file)
                            # if is_best[1]:
                            #     save_model(args, student_model, tokenizer, model_name='best_acc_pytorch_model.bin')
                            #     result['best_acc'] = best_dev_acc[1]
                            #     result_to_file(result, output_eval_file)
                    # if epoch_ == 6 and args.task_name.lower() == 'sst-2':
                    #     save_model(args, student_model, tokenizer, model_name=f'step_{str(global_step)}_pytorch_model.bin')

                    student_model.train()
            if 10 >= epoch_ >= 5 and args.task_name.lower() == 'sst-2':
                save_model(args, student_model, tokenizer, model_name=f'epoch_{str(epoch_)}_pytorch_model.bin')




if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    taskname = "SST-2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=f"../data/glue_data/{taskname}", type=str)
    parser.add_argument("--teacher_model", default=f"teacher/teacher_{taskname.lower()}",type=str)
    parser.add_argument("--student_model", default='/data/lxk/NLP/TinyBERT/zhh_emd/model/2nd_General_TinyBERT_4L_312D', type=str,)
    parser.add_argument("--task_name", default=taskname, type=str)

    parser.add_argument("--output_dir", default=f"../model/{taskname}/Layer6_EMD_T_", type=str)

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument('--eval_step', type=int, default=50)

    parser.add_argument('--pred_distill', action='store_true')
    parser.add_argument("--seperate", default=False, type=str2bool)
    parser.add_argument("--add_softmax", default=True, type=str2bool)

    parser.add_argument("--tinybert", default=False, type=str2bool)
    parser.add_argument("--pkd", default=False, type=str2bool)
    parser.add_argument("--use_emd", default=True, type=str2bool)
    parser.add_argument("--update_weight", default=True, type=str2bool)
    parser.add_argument("--one_step", default=True, type=str2bool)
    parser.add_argument("--new_pred_loss", default=True, type=str2bool)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--theta", type=float, default=1.0)

    # Not often modified
    parser.add_argument('--T', type=float, default=1.)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--do_eval", default=False, type=str2bool)
    parser.add_argument("--do_predict", default=False, type=str2bool)
    parser.add_argument("--use_att", default=True, type=str2bool)
    parser.add_argument("--use_rep", default=True, type=str2bool)
    parser.add_argument("--use_embedding", default=True, type=str2bool)

    # Generally don't care
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--emb_linear", default=False, type=str2bool)
    parser.add_argument("--no_pretrain", action='store_true')
    parser.add_argument("--use_init_weight", action='store_true')
    parser.add_argument("--share_param", default=True, type=str2bool)
    parser.add_argument("--is_conv", action='store_true')
    parser.add_argument("--do_lower_case", default=True, type=str2bool)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--aug_train', action='store_true')
    parser.add_argument('--data_url', type=str, default="")
    parser.add_argument('--tb_onestep', type=str, default="")
    parser.add_argument('--T_emd', type=float, default=1)
    parser.add_argument('--reduce_T', type=float, default=1.0)
    parser.add_argument('--is_teacher', type=str2bool, default=True)

    args=parser.parse_args()

    dir_index = 0

    if args.tinybert:
        args.one_step = False
        args.new_pred_loss = False
        args.use_emd = False
        args.learning_rate = 5e-05

    if args.pkd:
        args.student_model = args.teacher_model
        args.one_step = True
        args.new_pred_loss = True

    args.seed = random.randint(0, 100000)

    new_out = args.output_dir + "Model_" + args.student_model.split('/')[-1] + "_" + str(dir_index)
    while os.path.exists(new_out):
        dir_index += 1
        new_out = args.output_dir + "Model_" + args.student_model.split('/')[-1] + "_" + str(dir_index)
    args.output_dir = new_out
    os.makedirs(args.output_dir)

    best_parms = {
        "cola": [1, 1, 0.001],
        "mnli": [1, 1, 0.01],
        "mrpc": [1, 1, 0.01],
        "sst-2": [1, 1, 0.01],
        "sts-b": [1, 1, 0.005],
        "qqp": [1, 1, 0.005],
        "qnli": [1, 1, 0.01],
        "rte": [1, 1, 0.005]
    }
    default_params = {
        "cola": {"num_train_epochs": 50, "max_seq_length": 64},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128, "learning_rate": 2e-2},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
        "sst-2": {"num_train_epochs": 8, "max_seq_length": 64, "learning_rate": 1e-2},
        "sts-b": {"num_train_epochs": 30, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 10, "max_seq_length": 128, "learning_rate": 2e-2},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 20, "max_seq_length": 128}
    }
    args.max_seq_len = default_params[args.task_name.lower()]["max_seq_length"]
    # args.num_train_epochs = default_params[args.task_name.lower()]["num_train_epochs"]

    args.alpha = best_parms[taskname.lower()][0]
    # args.T = best_parms[taskname.lower()][1]
    # args.beta = best_parms[taskname.lower()][2]
    logger.info('The args: {}'.format(args))
    do_train(args)
