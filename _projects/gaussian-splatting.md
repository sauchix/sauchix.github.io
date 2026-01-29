---
layout: page
title: 3D Gaussian Splatting and Spherical Harmonics with Implementations
description: A deep dive into the math of splattin.
img: 
importance: 1
category: work
---

## Introduction

In this study, I aim to dive into the theory, mathematics and implementation of the 3D Gaussian Splatting technique, using the workflow of the original paper Kerbl 2023 (https://arxiv.org/pdf/2308.04079).

3D Gaussian Splatting (3DGS) has revolutionized real-time rendering, as 3D rednering speed from Neural Network Radiance Feild used to take hours to train. 3DGS is a rasterization technique, drawing the 3D points of the real world into 2D points in the projection plane, though the persepctive matrix p' = $\pi$ p. Therefore, a single rasterized Gaussian has these parameters:
- Color (Spherical Harmonics)
- Position
- Covariance
- Alpha (How transparent it is)

<b><u>Spherical Harmonics</u></b>

In a 3D gaussian the spherical harmonics coefficients take up almost 80% of the parameters, so it is important for us to understand how these coefficients come. 3DGS doesn't store a single RGB color, instead we want to be able to model the fact that color is a function of the viewing direction. This is where spherical harmonics comes in, a Fourier series function for colors on a sphere:
$$ f(\theta, \phi) = \sum_{\ell=0}^{L} \sum_{m=-\ell}^{\ell} c_{\ell m} Y_{\ell m}(\theta, \phi) $$
- f models view-dependent color
- $Y_{\ell m}$ is a set of patterns - fixed orthonormal basis functions on the unit sphere
- $C_{\ell m}$ are coefficients that weight the $Y_{\ell m} 

In Gaussian Splatting, we need to learn the coefficients of each coefficients for each Gaussian.

$Y_{\ell m}$ is a set of harmonic patterns which are mathematical constants we can pre-calculate. With a weighted sum of these standard patterns, we can reconstruct any complex lighting reflection. The $\sum_{\ell=0}^{L}$ summation models the frequency of the patterns, the details we are going to use. The higher the L the better, the more complex the color we can represent and the less errors in the approximation. However, this also means more memory and more compute. $\ell$ controls the frequency, how fined and detailed the color pattern is. $\sum_{m=-\ell}^{\ell}$ - When $\ell$ = 0, we have 1 term. When $\ell$ = 1, we have 3 terms from -1 to 1. If we have a complexity level of $\ell$, we need 2 $\ell$ + 1 different shapes to cover all orientations. $\ m$ controls the orientation, how pattern is rotated around the sphere.

In standard 3D Gaussian Splatting, we usually set the maximum degree L to 3. This means we will have 1 + 3 + 5 + 7 = 16 parameters. After doing this for each of R, G, B color channels, we will have in total of 48 spherical harmonics components we need to store to model the colors of the Gaussians, usually more than 80% of the parameters.

$Y_{\ell m}$ are usually represented using angles $\theta$ and $\phi$ 
$$
Y_{\ell m}(\theta, \phi) = N_{\ell m} P_{\ell}^{|m|}(\cos \theta) e^{im\phi}
$$
- $N_{\ell m}$ is the Normalisation constant - 
$$
N_{\ell m} = \sqrt{\frac{2\ell + 1}{4\pi} \frac{(\ell - |m|)!}{(\ell + |m|)!}}
$$
- $P_{\ell}^{|m|}(\cos \theta)$ is the Legendre Polynomial that defines the shape along the latitude.
- $e^{im\phi}$ is the Azimuthal phase that defines the shape along the Longitude.

In 3DGS, we can replace these complex terms with cartisian coordinates x, y and z: 
$$
x = \sin\theta \cos\phi, \quad y = \sin\theta \sin\phi, \quad z = \cos\theta
$$
Now each $Y_{\ell m}$ becomes a polynomial in (x,y,z), which will be used in our code implementations to represent our Spherical harmonics.

Degree 0: $$ Y_{0,0}(\theta, \phi) = \sqrt{\frac{1}{4\pi}} \quad \Rightarrow \quad Y_{0,0} = 0.28209479177387814 $$
Degree 1:
$$
\begin{aligned}
Y_{1,-1}(\theta, \phi) &= -\sqrt{\frac{3}{4\pi}} \sin\theta \sin\phi &&\Rightarrow Y_{1,-1} = -0.4886025119029199 \, y \\
Y_{1,0}(\theta, \phi) &= \sqrt{\frac{3}{4\pi}} \cos\theta &&\Rightarrow Y_{1,0} = +0.4886025119029199 \, z \\
Y_{1,1}(\theta, \phi) &= -\sqrt{\frac{3}{4\pi}} \sin\theta \cos\phi &&\Rightarrow Y_{1,1} = -0.4886025119029199 \, x
\end{aligned}
$$
etc...

The $C_{\ell m}$ will be learned paramters from Gaussian training, before combining them with the fixed basis functions Y.
