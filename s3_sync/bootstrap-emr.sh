#!/bin/bash
# install requirements on all cluster nodes, not only master
# update pip to prevent pyarrow installation error
sudo python3 -m pip install -U setuptools
sudo python3 -m pip install -U pip
sudo python3 -m pip install wheel
sudo python3 -m pip install pillow
# same version of tensorflow and keras to avoid module not found error
sudo python3 -m pip install tensorflow==2.11.0
sudo python3 -m pip install keras==2.11.0
sudo python3 -m pip install pandas==1.2.5
sudo python3 -m pip install pyarrow
sudo python3 -m pip install boto3
sudo python3 -m pip install s3fs
sudo python3 -m pip install fsspec
