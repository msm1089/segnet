# SegNet
[SegNet](http://mi.eng.cam.ac.uk/projects/segnet/) implementation in Tensorflow.

## Environments
- Python 3.7.7
- Numpy==1.18.4
- Pillow==7.1.2
- matplotlib==3.2.1
- tensorflow-gpu==2.1.0


## Usage  
1.	Clone the repo:
```
git clone https://github.com/msm1089/segnet.git
```

2. Download CamVid Dataset
```
bash download.sh
```

3. Convert CamVid dataset to TFRecord format
```
python camvid.py --target train
```

4. Create conda env (recommended but optional):
```
conda create -n tf-gpu
conda activate tf-gpu
```

Install tensorflow-gpu:
```
conda install tensorflow-gpu
```

Install Pillow & matplotlib:
```
pip install Pillow matplotlib
```

5. Training
```
python train.py \
  --iteration 20000 \
  --snapshot 4000 \
  --optimizer adadelta \
  --learning_rate 1.0
```

6. Evaluation
```
python eval.py \
  --resdir eval \
  --outdir output/camvid/segnet \
  --checkpoint_dir output/camvid/segnet/trained_model \
  --num_sample 233
```


## Results
<div align="center">
<img src="images/1.png">
</div>


## Reference
- https://github.com/Pepslee/tensorflow-contrib
- https://github.com/warmspringwinds/tf-image-segmentation
- https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation
- https://github.com/eragonruan/refinenet-image-segmentation
- https://github.com/DrSleep/tensorflow-deeplab-resnet
