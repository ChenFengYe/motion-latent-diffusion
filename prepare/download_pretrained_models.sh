#!/bin/bash
echo -e "The pretrained models will stored in the 'pretrained_models' folder\n"
gdown "https://drive.google.com/uc?id=1qU9YWJsvSuuWvivVV_wOj2_V60qGgVMM"

echo -e "Please check that the md5sum is: 201c4344ac70bce60035fdf46dc8c247"
echo -e "+ md5sum pretrained_models.tgz"
md5sum pretrained_models.tgz

echo -e "If it is not, please rerun this script"

sleep 3
tar xfzv pretrained_models.tgz

echo -e "Cleaning\n"
rm pretrained_models.tgz

echo -e "Downloading done!"
