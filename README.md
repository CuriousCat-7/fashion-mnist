# DistillNas
我知道代码很难读，但我懒得整理

train -> search -> compare with random sampled architerctures


# Zalando's MNIST fashion replacement
This repo is archived.

Zalando recently released an MNIST replacement. The issues with using MNIST are
known but you can read about the dataset and their motivation [here](https://github.com/zalandoresearch/fashion-mnist).

### Training
```
python train.py

        --model         # specify model, (FashionSimpleNet, resnet18)
        --patience      # early stopping
        --batch_size    # batch size
        --nepochs       # max epochs
        --nworkers      # number of workers
        --seed          # random seed
        --data          # MNIST or FashionMNIST
```


### Results
|   | FashionSimpleNet | ResNest18 |
| ------------- | ------------- |-----------|
| MNIST  | 0.994  | 0.994|
| FashionMNIST  | 0.923  | 0.920|


