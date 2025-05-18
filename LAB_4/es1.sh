# Model CNN2
python es1.py --cnn True --cnn_ty False --model_path_cnn "Models/CNN2_pretrain.pth" --temp 1000
python es1.py --cnn True --cnn_ty False --model_path_cnn "Models/CNN2_pretrain_aug_0.05.pth" --temp 1000
python es1.py --cnn True --cnn_ty False --model_path_cnn "Models/CNN2_pretrain_aug_0.1.pth" --temp 1000
python es1.py --cnn True --cnn_ty False --model_path_cnn "Models/CNN2_pretrain_aug_rand.pth" --temp 1000

# Model CNN
python es1.py --cnn True --cnn_ty True --model_path_cnn "Models/CNN_pretrain.pth" --temp 1000
python es1.py --cnn True --cnn_ty True --model_path_cnn "Models/CNN_pretrain_aug_0.05.pth" --temp 1000
python es1.py --cnn True --cnn_ty True --model_path_cnn "Models/CNN_pretrain_aug_0.1.pth" --temp 1000
python es1.py --cnn True --cnn_ty True --model_path_cnn "Models/CNN_pretrain_aug_rand.pth" --temp 1000

# Model AE
python es1.py --ae True --cnn False --model_path_ae "Models/AE_pretrain.pth"
python es1.py --ae True --cnn False --model_path_ae "Models/AE_pretrain_aug_0.05.pth"
python es1.py --ae True --cnn False --model_path_ae "Models/AE_pretrain_aug_0.1.pth"
python es1.py --ae True --cnn False --model_path_ae "Models/AE_pretrain_aug_rand.pth"







