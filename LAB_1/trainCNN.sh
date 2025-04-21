# CNN
python main.py --lr 1e-3 --MLP False --scheduler True --residual True --layers 2 2 2 2 --dataset "CIFAR10"
python main.py --lr 1e-3 --MLP False --scheduler False --residual True --layers 2 2 2 2 --dataset "CIFAR10"
python main.py --lr 1e-3 --MLP False --scheduler True --residual False --layers 2 2 2 2 --dataset "CIFAR10"
python main.py --lr 1e-3 --MLP False --scheduler False --residual False --layers 2 2 2 2 --dataset "CIFAR10"

python main.py --lr 1e-3 --MLP False --scheduler True --residual True --layers 3 4 6 3 --dataset "CIFAR10"
python main.py --lr 1e-3 --MLP False --scheduler False --residual True --layers 3 4 6 3 --dataset "CIFAR10"
python main.py --lr 1e-3 --MLP False --scheduler True --residual False --layers 3 4 6 3 --dataset "CIFAR10"
python main.py --lr 1e-3 --MLP False --scheduler False --residual False --layers 3 4 6 3 --dataset "CIFAR10"

python main.py --lr 1e-3 --MLP False --scheduler True --residual True --layers 5 6 8 5 --dataset "CIFAR10"
python main.py --lr 1e-3 --MLP False --scheduler False --residual True --layers 5 6 8 5 --dataset "CIFAR10"
python main.py --lr 1e-3 --MLP False --scheduler True --residual False --layers 5 6 8 5 --dataset "CIFAR10"
python main.py --lr 1e-3 --MLP False --scheduler False --residual False --layers 5 6 8 5 --dataset "CIFAR10"








