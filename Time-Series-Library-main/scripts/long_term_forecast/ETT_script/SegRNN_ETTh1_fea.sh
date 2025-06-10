export CUDA_VISIBLE_DEVICES=2

model_name=SegRNNfea

#mse:0.0522703155875206, mae:0.17826946079730988

seq_len=720 #96
for pred_len in 96
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1fea \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len 48 \
  --enc_in 1 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1
done
