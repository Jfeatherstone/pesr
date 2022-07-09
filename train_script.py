import os
import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow import keras

sys.path.append('/eno/jdfeathe/pepe')
import pepe
from pepe.simulate import genSyntheticResponse
from pepe.preprocess import circularMask

from IPython.display import clear_output
plt.rcParams["figure.dpi"] = 150

############################
# Params

dsFactor = 10

batchsize = 1000
noiseLevel = .03

modelPath = 'model'

# Set up the model
convArgs = {"activation": 'relu',
            "kernel_initializer": 'Orthogonal',
            "padding": 'same'}

############################

if os.path.exists(modelPath):
    model = keras.models.load_model(modelPath)
else:
    inputs = keras.Input(shape=(None,None,1))
    x = keras.layers.Conv2D(64, 5, **convArgs)(inputs)
    x = keras.layers.Conv2D(64, 3, **convArgs)(x)
    x = keras.layers.Conv2D(64, 3, **convArgs)(x)
    x = keras.layers.Conv2D(dsFactor**2, 3, **convArgs)(x)
    outputs = tf.nn.depth_to_space(x, dsFactor)

    model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='MSE')

# Use pepe to generate fake photoelastic images
def genTrainingImages(batchsize, radius=150, minForce=0, maxForce=1, maxAlpha=.25, minBeta=0, maxBeta=np.pi, minNumForces=2, maxNumForces=4, fSigma=100, pxPerMeter=1e4, brightfield=False, noiseLevel=.05):
    numForces = np.random.randint(minNumForces, maxNumForces+1, size=batchsize)
    forceArr = [np.random.uniform(minForce, maxForce, size=numForces[i]) for i in range(batchsize)]
    # Normal for this one, since these are generally 0
    alphaArr = [np.random.normal(0, maxAlpha/2, size=numForces[i]) for i in range(batchsize)]
    betaArr = [np.random.uniform(minBeta, maxBeta, size=numForces[i]) for i in range(batchsize)]

    imageArr = np.zeros((batchsize, 2*radius, 2*radius))

    for i in range(batchsize):
        imageArr[i] = genSyntheticResponse(forceArr[i], alphaArr[i], betaArr[i], fSigma, radius, pxPerMeter, brightfield, imageSize=(2*radius, 2*radius))

    threeChannelImageArr = np.zeros((batchsize, 2*radius, 2*radius, 3), dtype=np.float64)

    for i in range(batchsize):
        threeChannelImageArr[i,:,:,0] = circularMask((2*radius, 2*radius), np.array((radius, radius), dtype=np.float64), radius)[:,:,0] * 0.5
        threeChannelImageArr[i,:,:,1] = imageArr[i] * 0.8

        threeChannelImageArr[i] += np.random.normal(0, noiseLevel, size=(2*radius, 2*radius, 3))
        threeChannelImageArr[i][threeChannelImageArr[i] > 1] = 1
        threeChannelImageArr[i][threeChannelImageArr[i] < 0] = 0

    return threeChannelImageArr

def convertToLuminance(images):
    newImages = np.zeros(images.shape[:3])
    yChannel = 0

    for i in range(len(images)):
        newImages[i] = tf.image.rgb_to_yuv(images[i])[:,:,yChannel]

    return newImages

########################3
# Training

lossArr = []

print('Initialization complete! Beginning training...')

while True:
    # Generate the images
    imageArr = genTrainingImages(batchsize+1, noiseLevel=0)
    imageArr = convertToLuminance(imageArr)
    
    # Downscale and add some noise
    dsImageArr = np.float64((imageArr + np.random.normal(0, noiseLevel, size=imageArr.shape))[:,::dsFactor,::dsFactor])
    
    lossArr.append(model.train_on_batch(dsImageArr[:-1,:,:,None], imageArr[:-1,:,:,None]))
   
    # Plot some stuff
    fig, ax = plt.subplots(1, 4, figsize=(11, 3))
    
    ax[0].plot(lossArr)
    ax[0].set_yscale('log')
    ax[0].set_title('Loss')
    
    ax[1].imshow(dsImageArr[-1])
    ax[1].set_title('Network Input')
    
    ax[2].imshow(model(dsImageArr[-1:,:,:,None])[0])
    ax[2].set_title('Network Output')
    
    ax[3].imshow(imageArr[-1])
    ax[3].set_title('True Output')
    
    plt.savefig('current_result.png')
    plt.close(fig)

    # Save the model
    model.save(modelPath)

    # Print
    print(f'Current loss: {lossArr[-1]}')
