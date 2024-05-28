model_name_or_path=$1
task_name=$2
dataset_name=$3
lora=$4 # True or False
from_linearhead=$5 # True or False
seed=$6

method_name=ft
if [ "$lora" = "True" ]; then
    method_name=lora
fi
loss_option=normal
if [ "$from_linearhead" = "True" ]; then
    loss_option="lp"
fi

python3 run_ntk.py \
  --model_name_or_path $model_name_or_path \
  --task_name $task_name \
  --dataset_name $dataset_name \
  --lora $lora \
  --from_linearhead $from_linearhead \
  --loss_option normal \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --output_dir results/ntk/$model_name_or_path/$dataset_name/$method_name/$loss_option/seed_$seed/ \
  --overwrite_output_dir \
  --seed $seed \
  --model_seed $seed \
  --report_to none
