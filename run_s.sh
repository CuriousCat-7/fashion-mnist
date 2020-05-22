# run search
python search_distill.py \
    --model FashionComplexNetDistillNas \
    --load-path saved-models/FashionComplexNetDistillNas-distill-19.pth.tar\
    --teacher-model FashionComplexNet \
    --teacher-path saved-models/FashionComplexNet-run-14.pth.tar
