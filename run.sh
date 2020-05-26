#python train.py --data FashionMNIST --model FashionComplexNet --patience 6
#CUDA_VISIBLE_DEVICES=1 python train_fair.py --data FashionMNIST --model FashionComplexNetNas --patience 6
#CUDA_VISIBLE_DEVICES=4 python train_fair_distill.py --data FashionMNIST --patience 6 \
#    --model FashionComplexNetDistillNas \
#    --teacher-model FashionComplexNet\
#    --teacher-path saved-models/FashionComplexNet-run-14.pth.tar\

CUDA_VISIBLE_DEVICES=4 python train_fair_distill.py --data FashionMNIST --patience 6 \
    --model FashionComplexNetDistillSCARLETNas\
    --teacher-model FashionComplexNet\
    --teacher-path saved-models/FashionComplexNet-run-14.pth.tar\
