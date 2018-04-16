Deep Alignment Network: A convolutional neural network for robust face alignment
===
  此项目为论文《Deep Alignment Network: A convolutional neural network for robust face alignment》的Tensorflow版本实现，
  论文及相关资料请参阅论文作者的Github开源项目:[GitHub](https://github.com/MarekKowalski/DeepAlignmentNetwork)
  
Getting started
-------  
* Tensorflow 1.3.0
* OpenCV 3.1.0 or newer

Running Code
---
* Download Datasets
* run `dir /b/s/p/w *.jpg *.png > ImageList.txt` to generate image list (Windows)
* run `DataSetPre.py` , Make sure you have modified the path.
* run `DAN.py` , Make sure you have modified the path & variable `STAGE`.

TODO
---
* Rewrite custorm layers on Gpu FOR performance.
* Use Tensorflow New API.

Test Now! New version will commit soon! 
===

Forward : 4~5ms per Image on I7 6700 / GTX 1080 Ti
