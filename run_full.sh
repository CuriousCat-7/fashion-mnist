
GPU=4
model_name="FashionComplexNetDistillSCARLETRandomNas"
suffix=Random
supernet_path=saved-models/FashionComplex-$suffix.pth.tar
pop_path=saved-pops/${suffix}_pop.pkl

CUDA_VISIBLE_DEVICES=$GPU python train_fair_distill.py --data FashionMNIST --patience 6 \
    --model $model_name\
    --teacher-model FashionComplexNet\
    --teacher-path saved-models/FashionComplexNet-run-14.pth.tar\
    --output-path $supernet_path &&
CUDA_VISIBLE_DEVICES=$GPU python search_fair.py \
    --model $model_name\
    --load-path $supernet_path\
    --pop-path $pop_path &&
CUDA_VISIBLE_DEVICES=$GPU python compare_fair.py \
    --model $model_name\
    --pop-path $pop_path\
    --load-path $supernet_path\
    --use-pretrained \
    --suffix $suffix
