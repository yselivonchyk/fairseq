#!/bin/bash

ENV=${ENV:=/shared/pytorch_bert}
echo "ENV" $ENV

NODE_RANK=$1
BUCKET_CAP_MB=25
TOTAL_UPDATE=1500000 MAX_SENTENCES=16 UPDATE_FREQ=1 LOG_INTERVAL=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="ens5"
TRAIN_SUBSET=train
DATABIN=/shared/roberta_poc/dataset_combined/book_wikicorpus_en
DATABIN=/scratch/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin

export FI_PROVIDER="efa"
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/usr/local/mpi/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/c uda-10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo /shared/checkpoints/herring_$OMPI_COMM_WORLD_RANK
/shared/$ENV/bin/fairseq-train $DATABIN --train-subset $TRAIN_SUBSET \
  --save-dir /shared/checkpoints/herring_$OMPI_COMM_WORLD_RANK \
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
        --validate-interval 500 --save-interval-updates 300 \
  --no-epoch-checkpoints --bucket-cap-mb $BUCKET_CAP_MB \
  --distributed-world-size 1 \
  --distributed-backend "herring"
