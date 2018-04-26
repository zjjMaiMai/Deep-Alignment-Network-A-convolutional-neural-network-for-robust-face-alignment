Deep Alignment Network: A convolutional neural network for robust face alignment
===

This is a **Tensorflow** implementations of paper *"Deep Alignment Network: A convolutional neural network for robust face alignment"*.
You can see **Original implementation** [here](https://github.com/MarekKowalski/DeepAlignmentNetwork).

-----------------

## System

* **No** Windows !

Getting started
-------  
* Tensorflow 1.7.0
* OpenCV 3.1.0 or newer

Train Model
---
* Download Datasets.
* Put `images & pts` in `SAME` folder.
* Write mirror file. There is a 68 landmark mirror file. [download](https://pan.baidu.com/s/1Ln_i00DRulDlgHJ8CmIqAQ)
* Preprocess.
```shell
python preprocessing.py --input_dir=... --output_dir=... --istrain=True --repeat=10 --img_size=112 --mirror_file=./Mirror68.txt
```
* Train model.
```shell
python DAN_V2.py -ds 1 --data_dir=preprocess_output_dir --data_dir_test=...orNone -nlm 68 -te=15 -epe=1 -mode train
python DAN_V2.py -ds 2 --data_dir=preprocess_output_dir --data_dir_test=...orNone -nlm 68 -te=45 -epe=1 -mode train
```

Eval Acc
---
* Download Datasets for test.
* Put `images & pts` in `SAME` folder.
* Preprocess.
```shell
python preprocessing.py --input_dir=... --output_dir=... --istrain=False --img_size=112
```
* Eval model Acc.
```shell
python DAN_V2.py -ds 2 --data_dir=preprocess_output_dir -nlm 68 -mode eval
```

Results on 300W
---
* Speed : 4ms per Image on GTX 1080 Ti
* Err : `1.34 %` on 300W common subset(bounding box diagonal normalization).

Pre-trained Model
---
TODO:You can download pre-trained model [here](). This model trained on 300W dataset.
