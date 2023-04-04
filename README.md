## Environment
Cuda 11.7

```shell
conda create -n octr python=3.8
conda activate octr

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && pip install -v -e . && cd ..

git clone https://github.com/octree-nn/ocnn-pytorch.git
cd ocnn-pytorch && python setup.py install --user && cd ..
```

## Training

```shell
bash dist_train.sh configs/fcaf3d-ocnn-scannet.py 4
```