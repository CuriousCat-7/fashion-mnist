CUDA_VISIBLE_DEVICES=2 python compare_fair.py --model FashionComplexNetDistillNas \
    --pop-path saved-pops/distill_pop.pkl\
    --load-path saved-models/FashionComplexNetDistillNas-distill-46.pth.tar\
    --use-pretrained\
    --suffix distill_only

#CUDA_VISIBLE_DEVICES=0 python compare_fair.py --model FashionComplexNetNas \
#    --pop-path saved-pops/fair_pop.pkl --suffix FD_F \
#    --load-path saved-models/FashionComplexNetDistillNas-train-77.pth.tar\
#    --use-pretrained

