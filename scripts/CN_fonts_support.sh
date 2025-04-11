#!/bin/bash

# 为matplotlib安装中文字体，解决可视化乱码

mkdir ~/.fonts
mv fonts/* ~/.fonts
rm -rf ~/.cache/matplotlib/*
sudo apt-get install -y fontconfig
fc-cache -fv