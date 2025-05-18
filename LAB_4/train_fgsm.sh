# CNN and AE with fgsm augmentation

#python pretrain.py --lr 1e-4 --epochs 200 --train_cnn True --cnn_ty False --train_AE False --aug_fgsm True --epsilon 0.05 --exp_name "CNN2_aug_0.05"
#python pretrain.py --lr 1e-4 --epochs 50 --train_cnn True --cnn_ty True --train_AE False --aug_fgsm True --epsilon 0.05 --exp_name "CNN_aug_0.05"
python pretrain.py --lr 1e-4 --epochs 200 --train_cnn False --train_AE True --aug_fgsm True --epsilon 0.05 --exp_name "AE_aug_0.05"

#python pretrain.py --lr 1e-4 --epochs 200 --train_cnn True --cnn_ty False --train_AE False --aug_fgsm True --epsilon 0.1 --exp_name "CNN2_aug_0.1"
#python pretrain.py --lr 1e-4 --epochs 50 --train_cnn True --cnn_ty True --train_AE False --aug_fgsm True --epsilon 0.1 --exp_name "CNN_aug_0.1"
python pretrain.py --lr 1e-4 --epochs 200 --train_cnn False --train_AE True --aug_fgsm True --epsilon 0.1 --exp_name "AE_aug_0.1"

#python pretrain.py --lr 1e-4 --epochs 200 --train_cnn True --cnn_ty False --train_AE False --aug_fgsm True --rand_epsilon True --exp_name "CNN2_aug_rand"
#python pretrain.py --lr 1e-4 --epochs 50 --train_cnn True --cnn_ty True --train_AE False --aug_fgsm True --rand_epsilon True --exp_name "CNN_aug_rand"
python pretrain.py --lr 1e-4 --epochs 200 --train_cnn False --train_AE True --aug_fgsm True --rand_epsilon True --exp_name "AE_aug_rand"






