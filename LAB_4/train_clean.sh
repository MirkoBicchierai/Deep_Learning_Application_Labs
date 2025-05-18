# CNN and AE
python pretrain.py --lr 1e-4 --epochs 200 --train_cnn True --cnn_ty False --train_AE False --aug_fgsm False --exp_name "CNN2"
python pretrain.py --lr 1e-4 --epochs 50 --train_cnn True --cnn_ty True --train_AE False --aug_fgsm False --exp_name "CNN"
python pretrain.py --lr 1e-4 --epochs 200 --train_cnn False --train_AE True --aug_fgsm False --exp_name "AE"






