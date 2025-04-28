# Lora
python main.py --dataset "stanfordnlp/sst2" --lora true --batch_size 16 --lr 2e-5 --lora_alpha 32 --lora_rank 8 --epochs 5
python main.py --dataset "stanfordnlp/sst2" --lora true --batch_size 16 --lr 2e-5 --lora_alpha 64 --lora_rank 16 --epochs 5

python main.py --dataset "stanfordnlp/sst2" --lora true --batch_size 16 --lr 2e-4 --lora_alpha 32 --lora_rank 8 --epochs 5
python main.py --dataset "stanfordnlp/sst2" --lora true --batch_size 16 --lr 2e-4 --lora_alpha 64 --lora_rank 16 --epochs 5

