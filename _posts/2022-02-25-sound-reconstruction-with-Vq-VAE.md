---
title: 'Vq-VAE Methods for Sound Reconstruction 🎻'
date: 2022-02-25
permalink: /posts/2022/02/Complex-Valued-Neural-Net/
tags:
  - Sound Reconstruction
  - VAE
  - Vq-VAE
  - Vector Quantization Methods
  - Variational Auto Encoder
  - Auto Encoder
image: architecture.jpeg
---

This Repo Contains Implementation of Vq-VAE methods for sound reconstruction in Pytorch

![elijah-m-henderson-xgT3iQDIijU-unsplash](https://user-images.githubusercontent.com/53477752/155848248-49f62114-0ff2-4605-927c-6d15c1726264.jpg)



# Signal-Processing
Signal Processing with Python and Librosa


## Voice Reconstruction Using Vq-VAE

This notebook proposes a method on how to reconstruct speech using vq-vae which has been first introduced by [ Oord et. al](https://arxiv.org/abs/1711.00937).



![picture](https://drive.google.com/uc?export=view&id=1XFC5_OYnwge7g14jmKZqHaBwuajsVDyP)



## Vq-VAE vs VAE
Main difference between Vq-VAE & VAE is that VAE learns a continuous latent representation of a given dataset, but Vq-VAE learns a discrete latent representation of dataset.  



## Architecture


![picture](https://drive.google.com/uc?export=view&id=1hVMCRd4ZeBMM581iLa5ZKV1C-gT4IUsQ)


- At the begining, encoder takes a batch of images with input shape of $X:(n, h, w, c)$ and outputs $Z_{e}:(n, h, w, d)$

- Then vector quantization layer takes $Z_{e}$ and for each vector in $Z_{e}$ it selects the nearest vector from the codebook based on $L_{2}$ norm and outputs $Z_{q}$ 

- Finally decoder takes $Z_{q}$ and reconstructs the input $X$.

## Detailed View on Vq-VAE Architecture

![picture](https://drive.google.com/uc?export=view&id=1a---6D9Nzvf7VJpfR_zDGujXSpJwG6cJ)


- Reshaping: First of all we need to reshape input from $(n, h, w, d)$ to $(n \times h \times w, d)$ .

- Calculating Distances: For each of d-dimensional vectors, we calculate their distance from each $k$, d-dimensional vectors in codebook and get a matrix of $(n \times h \times w, k)$.

- Argmin: Next for each row of the matrix, we apply argmin function to get the nearest vector index from codebook and do one-hot encoding no each row (in fact the value of the nearest vector will be 1 and rest would be 0).

- Index from Codebook: After that we multiply the one-hotted matrix to the whole codebook and we get a matrix of $(n \times h \times w, d)$ dimension.

- Finally we reshape $(n \times h \times w, d)$ to $(n, h, w, d)$ and give it to the decoder to reconstruct the input data 



##  Some High Resolution Constructed Images


![picture](https://drive.google.com/uc?export=view&id=1rAicXUoCWrLEKtA-Wyvwj2ydNhsbQzk3)


### References
[1] https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a

[2] https://arxiv.org/pdf/1711.00937.pdf



<embed src="https://arxiv.org/pdf/1703.10135.pdf" type="application/pdf" width="800px" height="800px">.

<audio src="https://raw.githubusercontent.com/mehdihosseinimoghadam/mehdihmblog.github.io/blob/gh-pages/assets/img/LJ037-0171.wav" controls preload="metadata"></audio>

<audio src="https://raw.githubusercontent.com/mehdihosseinimoghadam/mehdihmblog.github.io/blob/gh-pages/assets/img/samples_1.wav" controls preload="metadata"></audio>






### Dataset:
  To train vq-vae model I used [speech commands tensorflow](https://www.kaggle.com/luantm/speech-commands-tensorflow) dataset, for simplicity I just used **Right**
  wav files, some wav samples with their spectrograms can be found below:
  
![wav1](https://user-images.githubusercontent.com/53477752/155848854-0b8e1b63-4384-464a-be2c-c54b81cdd852.png)


  <audio src="https://github.com/mehdihosseinimoghadam/mehdihosseinimoghadam.github.io/blob/master/_posts/Pitbull-Bon-Bon-320.mp3" controls preload="metadata"></audio>
  
  
<audio src="https://www.dropbox.com/s/mxhc3m62tg7y0gg/samples_1.wav" controls preload></audio>
<audio src="https://www.dropbox.com/home?preview=samples_1.wav" controls preload></audio>
  
  
  <iframe src="https://www.dropbox.com/s/mxhc3m62tg7y0gg/samples_1.wav?dl=0" allow="autoplay" style="display:none" id="iframeAudio"></iframe>

  
  <iframe src="https://www.dropbox.com/s/mxhc3m62tg7y0gg/samples_1.wav" allow="autoplay" style="display:none" id="iframeAudio"></iframe>

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/Complex_Deep_Neural_Network.ipynb)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)






## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

Released 2022 by [Mehdi Hosseini Moghadam](https://github.com/mehdihosseinimoghadam)
