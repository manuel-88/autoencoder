# autoencoder with Street View House Numbers(SVHN) Dataset

I build a simple autoencoder with tensorflow and Googles Street View House Numbers Dataset. I converted the RGB-Images to grayscale and scaled the pixel values between 0 and 1 floating point numbers. But the results I get are very poor even if I grow the amount of units in the hidden layer nearly to the size of the input array . I get much better results if I convert the Images to binary values.  

The upper row in each Picture are the input Images the lower row the output images
![ae_binary_svhn](https://cloud.githubusercontent.com/assets/14162105/19848244/750503ea-9f4c-11e6-8833-f4571801510f.png)
