export TASK_NAME=superglue
export DATASET_NAME=wic

bs=16
lr=1e-3
loss_option=normal
epoch=10
lora_alpha=8
lora_r=8


for seed in 1001 2002 3003 4004 5005
do
    python3 run.py \
      --model_name_or_path roberta-base \
      --task_name $TASK_NAME \
      --dataset_name $DATASET_NAME \
      --loss_option $loss_option \
      --do_train \
      --do_eval \
      --do_predict \
      --max_seq_length 128 \
      --per_device_train_batch_size $bs \
      --learning_rate $lr \
      --num_train_epochs $epoch \
      --output_dir results/evaluation/roberta/$DATASET_NAME/lora/$loss_option/seed_$seed/ \
      --overwrite_output_dir \
      --seed $seed \
      --model_seed $seed \
      --save_strategy epoch \
      --evaluation_strategy epoch \
      --load_best_model_at_end \
      --metric_for_best_model accuracy\
      --greater_is_better True \
    --save_total_limit 1 \
      --lora  \
      --lora_alpha $lora_alpha \
      --lora_r $lora_r \
    --report_to none
done
