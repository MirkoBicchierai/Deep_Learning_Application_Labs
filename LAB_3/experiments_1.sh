# Full Model
python main.py --dataset "rotten_tomatoes" --batch_size 16 --lr 2e-5 --epochs 10
python main.py --dataset "rotten_tomatoes" --batch_size 16 --lr 2e-4 --epochs 10

# Lora
python main.py --dataset "rotten_tomatoes" --lora true --batch_size 16 --lr 2e-5 --lora_alpha 32 --lora_rank 8 --epochs 10
python main.py --dataset "rotten_tomatoes" --lora true --batch_size 16 --lr 2e-5 --lora_alpha 64 --lora_rank 16 --epochs 10

python main.py --dataset "rotten_tomatoes" --lora true --batch_size 16 --lr 2e-4 --lora_alpha 32 --lora_rank 8 --epochs 10
python main.py --dataset "rotten_tomatoes" --lora true --batch_size 16 --lr 2e-4 --lora_alpha 64 --lora_rank 16 --epochs 10

