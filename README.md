# autoencoder with Street View House Numbers(SVHN) Dataset

I build a simple autoencoder with tensorflow and Googles Street View House Numbers Dataset. I converted the RGB-Images to grayscale and scaled the pixel values between 0 and 1 floating point numbers. But the results I get are very poor even if I grow the amount of units in the hidden layer nearly to the size of the input array . I get much better results if I convert the Images to binary values.  

The upper row in each Picture are the input Images the lower row the output images
![ae_binary_svhn](https://cloud.githubusercontent.com/assets/14162105/19847792/94680ece-9f49-11e6-91c1-8fddc00924b4.png)
Binary pixels values (thresholded) as input.

![ae_float_svhn](https://cloud.githubusercontent.com/assets/14162105/19847795/964f5256-9f49-11e6-8d9c-9389e0569267.png)
Same input images with floating points. All the outputs are the same noisy images.


Is it possible to get a greyscale output or is an autoencoder not able to learn Images with floating point pixel values? 

I uploaded the code on github
The "download_SVHN.py" file download the dataset and convert it to a matrice. With "autoencoder_SVHN.py" the autoencoder
get executed and plot the results. Line 40 and 41 convert the images to binary pixel values. If you uncomment it, the images remain as grayscale.
