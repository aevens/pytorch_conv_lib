import sys

import requests
import numpy as np
np.random.seed(42)

import pandas as pd
import os
import idx2numpy
import gzip
import torch
torch.manual_seed(42)

def mnist_downloader():
    train_features = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
    train_labels = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
    test_features = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
    test_labels = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"

    #Create storage directory
    temp_path = "./temp"
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    #Iterate through links and download if not already in place
    for web_link in [train_features, train_labels, test_features, test_labels]:

        zfile_suffix = web_link[web_link.find('/',20)+1:]
        zfilename = os.path.join(temp_path, zfile_suffix )

        if not os.path.exists(zfilename):
            resp = requests.get(web_link)
            zfile = open(zfilename, 'wb')
            zfile.write(resp.content)
            zfile.close()

    #Unzip and load into numpy arrays
    def load_data(filename):
        filename = os.path.join(temp_path, filename)
        f = gzip.open(filename)
        container = idx2numpy.convert_from_file(f)
        return container

    elems = sorted(os.listdir(temp_path), reverse=True)

    trainys = load_data(elems[0])[:50000].reshape(-1,1)
    trainxs = load_data(elems[1])[:50000]
    validys = load_data(elems[0])[50000:].reshape(-1,1)
    validxs = load_data(elems[1])[50000:]
    testys = load_data(elems[2]).reshape(-1,1)
    testxs = load_data(elems[3])
    
    #Transform x-data from 0-255 ints to 0 - 1.0 floats
    trainxs = trainxs.astype(dtype='float') / 255
    validxs = validxs.astype(dtype='float') / 255
    testxs = testxs.astype(dtype='float') / 255
    
    return (trainys,trainxs, validys, validxs, testys, testxs)