# Linear Evaluation
python main2.py --lr 1e-3 --optimizer "SGD" --scheduler True --num_layers 0 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-3 --optimizer "SGD" --scheduler False --num_layers 0 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "SGD" --scheduler True --num_layers 0 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "SGD" --scheduler False --num_layers 0 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-3 --optimizer "Adam" --scheduler True --num_layers 0 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-3 --optimizer "Adam" --scheduler False --num_layers 0 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "Adam" --scheduler True --num_layers 0 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "Adam" --scheduler False --num_layers 0 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
# Unfreeze the earliest 3 layer
python main2.py --lr 1e-3 --optimizer "SGD" --scheduler True --num_layers 3 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-3 --optimizer "SGD" --scheduler False --num_layers 3 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "SGD" --scheduler True --num_layers 3 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "SGD" --scheduler False --num_layers 3 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-3 --optimizer "Adam" --scheduler True --num_layers 3 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-3 --optimizer "Adam" --scheduler False --num_layers 3 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "Adam" --scheduler True --num_layers 3 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "Adam" --scheduler False --num_layers 3 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
# Unfreeze the earliest 5 layer
python main2.py --lr 1e-3 --optimizer "SGD" --scheduler True --num_layers 5 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-3 --optimizer "SGD" --scheduler False --num_layers 5 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "SGD" --scheduler True --num_layers 5 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "SGD" --scheduler False --num_layers 5 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-3 --optimizer "Adam" --scheduler True --num_layers 5 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-3 --optimizer "Adam" --scheduler False --num_layers 5 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "Adam" --scheduler True --num_layers 5 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2
python main2.py --lr 1e-2 --optimizer "Adam" --scheduler False --num_layers 5 --path "Models/8-CNN-Residual-Scheduler.pth" --layers 2 2 2 2