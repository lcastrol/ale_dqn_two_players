sudo apt-get install python-matplotlib python-opencv libopenblas-dev
sudo apt-get install build-essential gfortran libatlas-base-dev

sudo pip install scipy
sudo pip install numpy

#lasagne/Theano
sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
sudo pip install Lasagne==0.1

#TensorFlow
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Install from sources" below.
#$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl

sudo pip install --upgrade $TF_BINARY_URL

