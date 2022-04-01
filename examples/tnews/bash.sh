accelerate launch tnews.py \
    --model_name_or_path junnyu/ChineseBERT-base \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --seed 42 \
    --output_dir tnews_outputs \
    --num_train_epochs 3
