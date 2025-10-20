# !/bin/sh

wget https://raw.githubusercontent.com/phoebetronic/mnist/refs/heads/main/mnist_train.csv.zip
wget https://raw.githubusercontent.com/phoebetronic/mnist/refs/heads/main/mnist_test.csv.zip
unzip mnist_train.csv.zip
rm -rf __MACOSX/ mnist_train.csv.zip
unzip mnist_test.csv.zip
rm -rf __MACOSX/ mnist_test.csv.zip