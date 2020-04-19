import PacMan_functionality as PacMan

import numpy as np
import matplotlib.pyplot as plt
import pptk

# TODO Sort out imports when CNN architecture is finished.
# TODO replace with import keras then replace from imports
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from datetime import datetime


# Call startup_scene() to load the initial game scene
global_cloud, spheres_collected = PacMan.startup_scene()

# View our pointcloud if we want
v = pptk.viewer(global_cloud['Positions'], global_cloud['Colors']/255) # Fast, requires pptk
#PacMan.show_point_cloud(global_cloud) # Slow, requires matplotlib

# Train a classification model to perform binary classification of the patch into whether it contains a sphere.
# Just Load model for now
from tensorflow import keras
model = keras.models.load_model('CNN_Model.h5')

# Define function sliding window
# This function goes through windows in an image and returns the x,y coordinates of the start
# of the windows containing spheres, and distance to sphere as a np array
def sliding_window_classifier( step_size, image, depth ):
    #image is (160, 240, 3)
    #window is (51, 51, 3)
    image_hight = image.shape[0] # 160
    image_width = image.shape[1] # 240
    window_size = 51
    result = []

    for y in range(0, image_hight-window_size, step_size): 
        for x in range (0, image_width-window_size, step_size):

            window = image[y:y+window_size, x:x+window_size, :]

            #Normaly you predict on an array of value, but here since we want the (x,y) coordinates this is easier
            window = np.expand_dims(window, axis=0) # Adds first dimension to data (1,51,51,3) -> (1,51,51,3)
            prediction = model.predict( np.array( window ) )
            prediction = prediction[0][0] #  Unpack prediction [[1.]] -> 1

            # Only want patches we are 100% sure contain spheres
            if (prediction == 1):                
                print(f'Middle pixel Depth: {depth[y+25, x+25]}')
                print(f'Coordinates of sphere: {[mapx[y+25, x+25], mapy[y+26, x+26], mapz[y+26, x+26]]}')

                plt.figure(figsize=(2,1))
                plt.title('Image')
                plt.imshow( window[0,:,:,:] )   
        
                # Add 'crosshair' to show middle pixel
                ax=plt.gca() 
                ax.spines['left'].set_position('center')
                ax.spines['bottom'].set_position('center')
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                plt.show()

                # Spheres are always in the centre of the window.
                # Middle depth is the depth of the pixel in the centre of the sphere and window.
                middle_depth = depth[y+25, x+25]

                # Need these to be returned so we can move to the closest sphere.
                # x, y are now the coordinates of the centre of the sphere.
                result.append( (x+25, y+25, middle_depth) )

    # Results are np array with columns: x, y, depth
    # If length is 0 then no spheres were found
    return np.asarray(result)   

# Not necessary but nicely prints out the current viewpoints
def visualise_maps(image, mapx, mapy, mapz, depth):
    # Inputs can be full 160x240 or 51x51 in size
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20,3))

    ax[0].set_title('image')
    im0 = ax[0].imshow(image)
    #fig.colorbar(im, ax=ax[0])

    ax[1].set_title('mapx')
    im1 = ax[1].imshow(mapx, cmap='bwr')
    fig.colorbar(im1, ax=ax[1])

    ax[2].set_title('mapy')
    im2 = ax[2].imshow(mapy, cmap='bwr')
    fig.colorbar(im2, ax=ax[2])

    ax[3].set_title('mapz')
    im3 = ax[3].imshow(mapz, cmap='bwr')
    fig.colorbar(im3, ax=ax[3])

    ax[4].set_title('depth')
    im4 = ax[4].imshow(depth, cmap='bwr')
    fig.colorbar(im4, ax=ax[4])
    plt.show()
    print()

# Initialise position and angle variable 
position = np.zeros([3])
angle = np.zeros([3])

print('Start time:')
print(datetime.now().strftime('%H:%M:%S') ) # Used to time the execution of the program.

window_step_size = 10   # How big the steps are between windows
rotation_angle = np.pi*1/6 # will rotate

while not np.all(spheres_collected): # While there are spheres to find
    # Get current image from viewpoint
    print('Getting maps.... may take a while')
    image, mapx, mapy, mapz, depth = PacMan.project_pointcloud_image(global_cloud, angle, position)
    visualise_maps(image, mapx, mapy, mapz, depth)

    # Extract patches from the scene
    # Extract features from the patches    
    # Predict the probability of a pixel being a sphere, based on the patch
    # Use probabilities to find sphere coordinates in 3D
    sphere_windows = sliding_window_classifier(window_step_size, image, depth)

    # Update camera appropriately
    if ( len(sphere_windows) == 0):
        #rotate
        print('### No spheres found, rotating')
        angle = angle - np.asarray([0, rotation_angle, 0])   # numpy - is an elementwise operation 

    else:   # Spheres have been detected
        # Find the location of the window with lowest depth, which will show the closest sphere
        # sphere_windows has columns (x, y, depth)
        min_index = np.argmin( sphere_windows[:,2] )# Get the index of the closest sphere 
        x = int( sphere_windows[min_index, 0] )     # Get x location of start of sphere
        y = int( sphere_windows[min_index, 1] )    

        # Get spacial coordinates of new sphere
        newx = mapx[y, x] # I want Y, X not X, Y 
        newy = mapy[y, x] 
        newz = mapz[y, x] 

        # Move position and update scene
        print(f'I am at position {position}')
        print(f'I have found {np.sum(spheres_collected)} out of {len(spheres_collected)} spheres!\n')

        position = np.asarray([newx, newy, newz])
        print(f'I am now at position {position}')
        
        # Update scene
        global_cloud, spheres_collected = PacMan.update_scene(position, spheres_collected);
        print(f'I have found {np.sum(spheres_collected)} out of {len(spheres_collected)} spheres!\n')
