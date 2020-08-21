#!/bin/bash

set -e

GB=450; let PWMB=$GB*1024/10; for i in {1..10}; do python -c "x=($PWMB*1024*1024/8)*(0,); import time; time.sleep(10*3600*24)" & echo "started" $i ; done


ENV=/shared/pytorch_bert
NODE_RANK=$1
BUCKET_CAP_MB=25
TOTAL_UPDATE=1500000 MAX_SENTENCES=16 UPDATE_FREQ=1 LOG_INTERVAL=1
export NCCL_DEBUG=INFO
export FI_PROVIDER="efa"
export NCCL_SOCKET_IFNAME="ens5"
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/usr/local/mpi/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
TRAIN_SUBSET=train
DATABIN=/shared/roberta_poc/dataset_combined/book_wikicorpus_en
DATABIN=/scratch/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin

# DATABIN=/shared/roberta_data/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin


$ENV/bin/python3 -m torch.distributed.launch \
--nproc_per_node=8 --nnodes=8 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=`head -n 1 /shared/fairseq_hosts` --master_port=12349 \
$ENV/bin/fairseq-train $DATABIN --train-subset $TRAIN_SUBSET \
  --save-dir /shared/checkpoints/fair_0 \
  --memory-efficient-fp16 --fp16 \
  --fast-stat-sync --task masked_lm \
  --criterion masked_lm --arch roberta_large \
        --sample-break-mode complete --tokens-per-sample 512 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
        --clip-norm 1.0 --lr-scheduler polynomial_decay --lr 0.0001 \
  --warmup-updates 15000 --total-num-update $TOTAL_UPDATE \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATE --skip-invalid-size-inputs-valid-test \
  --log-format json --log-interval $LOG_INTERVAL --seed 7 \
        --validate-interval 5000 --save-interval-updates 3000 \
  --no-epoch-checkpoints --bucket-cap-mb $BUCKET_CAP_MB \
  --distributed-no-spawn
