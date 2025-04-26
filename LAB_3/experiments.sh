# Full Model
python main.py --batch_size 16 --lr 2e-5
python main.py --batch_size 16 --lr 2e-4

# Lora
python main.py --lora true --batch_size 16 --lr 2e-5 --lora_alpha 32 --lora_rank 8
python main.py --lora true --batch_size 16 --lr 2e-5 --lora_alpha 64 --lora_rank 16

python main.py --lora true --batch_size 16 --lr 2e-4 --lora_alpha 32 --lora_rank 8
python main.py --lora true --batch_size 16 --lr 2e-4 --lora_alpha 64 --lora_rank 16

