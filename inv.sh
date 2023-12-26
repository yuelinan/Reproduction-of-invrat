python invrat.py \
  --gpu_id=0 \
  --data_dir /movie \
  --dataset movie_reviews \
  --gradient_accumulation_steps=1 \
  --model_name=INVRAT \
  --lr=2e-05 \
  --max_len=512 \
  --alpha_rationle=0.1 \
  --types=train \
  --epochs=30 \
  --batch_size=4 \
  --class_num=2 \
  --save_path=/output \
  --alpha=0.03 \
  --beta=0.01 \
  --seed=1 \
  --is_da=no \
  --pretraining=yes \

