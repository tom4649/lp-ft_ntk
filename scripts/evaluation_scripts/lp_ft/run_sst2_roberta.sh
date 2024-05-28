export TASK_NAME=glue
export DATASET_NAME=sst2

bs=8
lr=1e-5
loss_option=lp_ft
epoch=10


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
    --output_dir results/evaluation/roberta/$DATASET_NAME/ft/$loss_option/seed_$seed/ \
    --overwrite_output_dir \
    --seed $seed \
    --model_seed $seed \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model accuracy\
    --greater_is_better True \
    --save_total_limit 1 \
      --report_to none
done
