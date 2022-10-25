import numpy
import scipy.io as io
import pywt # Wavelet
from scipy.fft import fft, fftfreq #Fourier
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm # for print progress 
import timeit
from scipy.signal import butter, lfilter # Butterworth Bandpass

#---------------------------------- explicit function to compute column wise sum
def colSum (arr):
    n = len(arr[0])
    m = len(arr)
    L = []
    for i in range(n):
        su = 0;
        for j in range(m):
            su += arr[j][i]
        L.append(su)
    return L

#---------------------------- for a 3D array Remove all values out for [0,1] values (have to check this more) 
def Cut_Values_Outside_Bandwidth_3D(array):
    for i in (range(0,array.shape[0])):
        for j in range(0,array.shape[1]):
            for k in range(0,array.shape[2]):
                if(array[i,j,k]>=1):
                    array[i,j,k] = 1
                if(array[i,j,k]<=0):
                    array[i,j,k] = 0
    return array

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------- The following functions resize Arrays/Images ----------------------------------------------#
#---------------------------- Resize a 2D Array to the disered size
def Resize_Img(image, width=None, height=None):
    dim=None
    (h,w) = image.shape[:2]
    if width is None and height is None:
        return image
    elif width is not None and height is not None:
        dim = (width, height)
    elif width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))
    resized = resize(image, dim)
    return resized

#---------------------------- Resize a list of 2D Arrays Images to the disered size
def Resize_list_of_Imges(arrays,width, height):
    resized_arrays = []
    for img in (arrays):
        resized_arrays.append(Resize_Img(img,width,height))
    resized_arrays = numpy.array(resized_arrays)
    return resized_arrays

