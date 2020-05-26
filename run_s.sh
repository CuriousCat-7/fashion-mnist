# run search
#CUDA_VISIBLE_DEVICES=2 python search_distill.py \
#    --model FashionComplexNetDistillNas \
#    --load-path saved-models/FashionComplexNetDistillNas-distill-46.pth.tar\
#    --teacher-model FashionComplexNet \
#    --teacher-path saved-models/FashionComplexNet-run-14.pth.tar\
#    #--load-path saved-models/FashionComplexNetDistillNas-distill-19.pth.tar\
#    #--load-path saved-models/FashionComplexNetDistillNas-distill-28.pth.tar\

#CUDA_VISIBLE_DEVICES=0 python search_fair.py \
#    --model FashionComplexNetNas \
#    --load-path saved-models/FashionComplexNetDistillNas-train-77.pth.tar\

    #--load-path saved-models/FashionComplexNetDistillNas-train-76.pth.tar\
    #--load-path saved-models/FashionComplexNetDistillNas-train-73.pth.tar

CUDA_VISIBLE_DEVICES=0 python search_fair.py \
    --model FashionComplexNetDistillSCARLETNas \
    --load-path saved-models/FashionComplexNetDistillSCARLETNas-train-3.pth.tar\
