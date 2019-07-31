# Image-colorization
This application is related to image processing based on CNN (Convolutional Neural Network). The basic idea behind this project is to convert black and white images to the colored image. I am using Convolutional Neural Network capable of colorizing black and white images.
Image colorization is the process of taking an input grayscale (black and white) image and then producing an output colorized image that represents the semantic colors and tones of the input.

I have started with the ImageNet dataset and converted all images from the RGB color space to the Lab color space.
Similar to the RGB color space, the Lab color space has three channels. But unlike the RGB color space, Lab encodes color information differently:

    The L channel encodes lightness intensity only
    The a channel encodes green-red.
    And the b channel encodes blue-yellow


The main concept behind this colorization is:

    Convert all training images from the RGB color space to the Lab color space.
    Use the L channel as the input to the network and train the network to predict the ab channels.
    Combine the input L channel with the predicted ab channels.
    Convert the Lab image back to RGB.
