CUDA_VISIBLE_DEVICES=2 python compare_fair.py --model FashionComplexNetDistillNas \
    --pop-path saved-pops/distill_pop.pkl
#CUDA_VISIBLE_DEVICES=3 python compare_fair.py --model FashionComplexNetNas \
#    --pop-path saved-pops/fair_pop.pkl

