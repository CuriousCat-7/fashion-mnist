#python train.py --data FashionMNIST --model FashionComplexNet --patience 6
#CUDA_VISIBLE_DEVICES=1 python train_fair.py --data FashionMNIST --model FashionComplexNetNas --patience 6
python train_distill.py --data FashionMNIST --patience 6\
    --model FashionComplexNetDistillNas --teacher-model FashionComplexNet\
    --teacher-path saved-models/FashionComplexNet-run-14.pth.tar
