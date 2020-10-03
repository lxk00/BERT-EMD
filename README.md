# BERT-EMD
Implementation of paper "BERT-EMD: Many-to-Many Layer Mapping for BERT Compression with Earth Mover's Distance" which is accepted by EMNLP2020.

Requirements:

`update soon`

# Data Prepare
Using `pregenerate_training_data.py` to produce the corpus of json format
```
# ${BERT_BASE_DIR}$ includes the BERT-base teacher model.

cd bert-emd
python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \ 
                  --bert_model ${BERT_BASE_DIR}$ \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$ 
```

# Distillation Step
For the student model,
We use the TinyBERT pretrained model as the student model, which can be access from [here](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT). The 2nd version models are used in our paper.


```
 # ${STUDENT_CONFIG_DIR}$ includes the config file of student_model.


cd bert-emd
python emd_task_distill.py --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${GENERAL_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \ 
                       --output_dir ${TMP_TINYBERT_DIR}$ \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 10 \
                       --aug_train \
                       --do_lower_case  
```
# Evaluation

The `task_distill.py` also provide the evalution by running the following command:

```
${TINYBERT_DIR}$ includes the config file, student model and vocab file.

python task_distill.py --do_eval \
                       --student_model ${TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${OUTPUT_DIR}$ \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128  
```
