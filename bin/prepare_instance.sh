#! /bin/bash
#
# This script is intended to get you up and running quickly
# on an EC2 spot block for deep learning

set -e

# MOTD said to run this
echo Activating tensorflow_p36 as per MOTD
source activate tensorflow_p36

# Install apps I want
echo install additional tools via apt-get
sudo apt-get -y install htop

# Clone the keras repo
echo clone the keras github repo
git clone git@github.com:keras-team/keras

# Install keras via pip
echo Install keras via pip3
pip3 install --user keras
