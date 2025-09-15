git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
sudo apt install libjpeg-turbo8 libjpeg-turbo-progs libosmesa6 libosmesa6-dev mesa-utils-extra libgl1-mesa-dri

apt install libglvnd0 libegl1 libgles2
apt install -y libegl-dev libgles2-mesa-dev
apt install -y mesa-utils-extra

pip install -U openmim
mim install "mmcv-lite"
mim install "mmdet"
mim install "mmpose"

