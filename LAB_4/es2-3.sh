# Model CNN
python es2-3.py --cnn True --cnn_ty True --ae False --model_path_cnn "Models/CNN_pretrain.pth"
python es2-3.py --cnn True --cnn_ty True --ae False --model_path_cnn "Models/CNN_pretrain_aug_0.05.pth"
python es2-3.py --cnn True --cnn_ty True --ae False --model_path_cnn "Models/CNN_pretrain_aug_0.1.pth"
python es2-3.py --cnn True --cnn_ty True --ae False --model_path_cnn "Models/CNN_pretrain_aug_rand.pth"

# Model CNN2
python es2-3.py --cnn True --cnn_ty False --ae False --model_path_cnn "Models/CNN2_pretrain.pth"
python es2-3.py --cnn True --cnn_ty False --ae False --model_path_cnn "Models/CNN2_pretrain_aug_0.05.pth"
python es2-3.py --cnn True --cnn_ty False --ae False --model_path_cnn "Models/CNN2_pretrain_aug_0.1.pth"
python es2-3.py --cnn True --cnn_ty False --ae False --model_path_cnn "Models/CNN2_pretrain_aug_rand.pth"

# Model AE
python es2-3.py --cnn False --ae True --model_path_ae "Models/AE_pretrain.pth"
python es2-3.py --cnn False --ae True --model_path_ae "Models/AE_pretrain_aug_0.05.pth"
python es2-3.py --cnn False --ae True --model_path_ae "Models/AE_pretrain_aug_0.1.pth"
python es2-3.py --cnn False --ae True --model_path_ae "Models/AE_pretrain_aug_rand.pth"







