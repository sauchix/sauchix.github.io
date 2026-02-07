---
layout: page
title: 3D Gaussian Splatting Maths and Theory
description: A deep dive into the math of splattin.
img: 
importance: 1
category: work
bibliography: papers.bib
---

## Introduction

In this study, I aim to dive into the theory and mathematics of the 3D Gaussian Splatting technique, using the main implementations from the original gaussian splatting paper {% cite kerbl20233dgs %}.

3D Gaussian Splatting (3DGS) has revolutionized real-time rendering, rednering speed from Neural Radiance Feild used to take hours to train, while 3DGS offers real-time performance. 3DGS is a rasterization technique, drawing the 3D points of the real world into 2D points in the projection plane, though a persepctive matrix p' = $\pi$ p (Jacobian matrix J is used in 3DGS instead for a linear approximation of camera projection). Therefore, a single rasterized Gaussian has these parameters {% cite huggingface_3dgs %}:
- Color (Spherical Harmonics coefficients)
- Position
- Covariance
- Alpha (How transparent it is)

3DGS are differentiable volumentric representation, which we can use backpropagation to train, as opposed to traditional geometry like meshes and point clouds.

## Spherical Harmonics

In a 3D gaussian the spherical harmonics coefficients take up almost 80% of the parameters {% cite papers100lines_sh %}, so it is important for us to understand how these coefficients come. 3DGS doesn't store a single RGB color, instead we want to be able to model the fact that color is a function of the viewing direction. This is where spherical harmonics comes in, a Fourier series function for colors on a sphere:
$$ f(\theta, \phi) = \sum_{\ell=0}^{L} \sum_{m=-\ell}^{\ell} c_{\ell m} Y_{\ell m}(\theta, \phi) $$
- f models view-dependent color
- $Y_{\ell m}$ is a set of patterns - fixed orthonormal basis functions on the unit sphere
- $C_{\ell m}$ are coefficients that weight the $Y_{\ell m}$

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

- $P_{\ell}^{\|m\|}(\cos \theta)$ is the Legendre Polynomial that defines the shape along the latitude.
- $e^{im\phi}$ is the Azimuthal phase that defines the shape along the Longitude.

In 3DGS, we can replace these complex terms with cartisian coordinates x, y and z: 
$$
x = \sin\theta \cos\phi, \quad y = \sin\theta \sin\phi, \quad z = \cos\theta
$$

Now each $Y_{\ell m}$ becomes a polynomial in (x,y,z), which will be used in our code implementations to represent our Spherical harmonics.

Degree 0:

$$ Y_{0,0}(\theta, \phi) = \sqrt{\frac{1}{4\pi}} \quad \Rightarrow \quad Y_{0,0} = 0.28209479177387814 $$

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


## Initialisation - COLMAP (Incremental SfM)

Before 3DGS starts learning, it needs a sparse point cloud to guess where the objects are, produced by a set of images of a static scene with corresponding cameras calibrated by SfM (Structure from Motion). Standard 3DGS uses COLMAP {% cite schoenberger2016sfm %}, an incremental Structure-from-Motion pipeline, which is a software wrapper of above, to figure out where the cameras are located when the photos are taken. 
- Camera Extrinsics & Intrinsics - Locations of the cameras and lens properties
- Sparse point Clouds - Starting positions x, y, z of the Gaussians centres

COLMAP uses 5-Point RANSAC Algorithm for efficiency to estimate the essential matrix, assuming the cameras are calibrated. This will repeat thousands of times and remove the bad data and outliers.

From these points, we create a set of 3D Gaussians defined by position(mean, $\mu$), covariance matrix ($\Sigma$), opacity ($\alpha$) and component color which is the spherical harmonics we mentioned(SH).

Gaussians basis function also defines the intensity with a full 3D covariance matrix $\Sigma$ defined in the world space {% cite zwicker2001surface %} centered at point $\mu$. This Gaussian is multiplied by $\alpha$ during the blending process.

$$
G(x) = e^{-\frac{1}{2} x^T \Sigma^{-1} x}
$$

- x - The query point, the specific coordinate in the 3D space (x, y, z) where we are trying to measure the density
- $\Sigma$ - Sigma, the covariance matrix (shape). In 3D the covariance bell curve has Scale and Rotation
- $\Sigma^{-1}$ - Inverse Covaraince which acts for stretching and rotating instructions
- $x^T \Sigma^{-1} x$ - This terms represents the distance check, which is how far away is x from the centre relative to the shape of the blob


## Optimization - Forward and Backward Procedure

As I mentioned 3DGS are differentiable representation, we can use backpropagation to train, and the optimzation is further improved by interleaving with adaptive density control steps which include cloning, splitting and removal. The goal is to allow high quality novel view synthesis for the end result. 

During Forward Pass we need to project 3D Gaussians to 2D for rendering {% cite zwicker2001surface %}, given viewing transformation W, the 2D covariance matrix $\Sigma'$ in camera coordinates is

$$
\Sigma' = JW \Sigma W^T J^T
$$

- J - Jacobian matrix provides the affine approximation of the projective transformation, which is also the first-order derivative that represents the second term of the Taylor series: $f(a)+J(x−a)$. This removes the non-linear warping of space to ensure a 3D Gaussian remains a 2D Gaussian after it is projected as Jacobian only includes the linear approximation of the zoom/stretch. Flattening the complex perspective matrix into a simple linear step also allows the GPU to handle it instantly.
- W - The viewing transformation
- $\Sigma$ - The 3D covarianace (shape)

The result $\Sigma'$ will still be a 3x3 matrix, skipping the third row and column of $\Sigma'$, we obtain the 2x2 variance matrix for the 2D ellipse we want to be projected.

During optimization, $\Sigma$ must be postive semi-definite to have phyiscal meaning (it can't have negative width). Therefore we always store the covariance matrix  $\Sigma$ of a 3D Gaussian with a scaling matrix S and rotation matrix R:
$$ 
\Sigma = RSS^{T} R^{T}
$$
The squared of the scale matrix ($SS^{T}$) ensures it is always positive. Kerbl also uses quaterions q - 4-number way to represent 3D rotation, while making sure q is normalised to obtain valid unit quaternion. The formulation also represents an aniotropic covariance, as the 3D Gaussian can have different 'spreads' along different axes. This allows the 3D Gaussians to adapt to geometry of different shapes of different scenes.

Backpropagation - The goal is to find how the change of 3D scaling s or 3D rotation q affects the shape of the 2D ellipse. We can apply chain rule to find the derivatives of scaling and rotation:

$$
\frac{d\Sigma'}{ds} = \frac{d\Sigma'}{d\Sigma} \frac{d\Sigma}{ds}
$$

and 

$$
\frac{d\Sigma'}{dq} = \frac{d\Sigma'}{d\Sigma} \frac{d\Sigma}{dq}
$$

- where $\frac{d\Sigma'}{ds}$ represents the scaling gradient, how much does the 2D covariance (shape) changes with the change in s (scale) and $\frac{d\Sigma'}{dq}$ represents the rotation gradient, how much does the shape changes with the change in q (rotations in quaternions).

Starting with finding $\frac{d\Sigma'}{d\Sigma}$:

Let U = JW, we get $\Sigma' = U \Sigma U^T$, combining the viewing trnasformation W and the Jacobian perspective project transformation J, we get a single matrix U that represents the linearized pipeline of transforming the 3D covariance into a 2D covariance.

$\Sigma'$ is the symmetric upper left 2x2 matrix of $U \Sigma U^T$. Since it is symmetric, the off-diagonal elements of the 2x2 matrix are the same, so it reduces the computational calculations from 4 to 3 unique multiplications $U_{1,i} U_{1,j}, U_{1,i} U_{2,j} and U_{2,i} U_{2,j}$.

We can find the partial derivatives 

$$
\frac{\partial \Sigma'}{\partial \Sigma_{i,j}} = \begin{bmatrix} 
U_{1,i} U_{1,j} & U_{1,i} U_{2,j} \\
U_{1,j} U_{2,i} & U_{2,i} U_{2,j}
\end{bmatrix}
$$

This gives us the weights (or gradients) of the affect of changinig the 3D covariance $\Sigma$ on the 2D covariance $\Sigma'$, which is the rows i and columns j of the U matrix. By focusing only on the 2x2 partial derivatives, it is way more computationally effective than doing the full 3x3 matrix multiplcations.

Nex we seek derivatives $\frac{d\Sigma}{ds}$ and $\frac{d\Sigma}{dq}$:

Since $\Sigma = RSS^{T} R^{T}$, let M = RS and $\Sigma = MM^T$

$$
\frac{d\Sigma}{ds} = \frac{d\Sigma}{dM} \frac{dM}{ds}
$$

and 

$$
\frac{d\Sigma}{dq} = \frac{d\Sigma}{dM} \frac{dM}{dq}
$$

Since covariance $\Sigma$ is symmetric, $\frac{d\Sigma}{dM} = 2M^T$, the derivative is essentially 2M.

For $\frac{dM}{ds}$, $M = RS$

$$M = \underbrace{\begin{bmatrix}
R_{11} & R_{12} & R_{13} \\
R_{21} & R_{22} & R_{23} \\
R_{31} & R_{32} & R_{33}
\end{bmatrix}}_{\text{Rotation } R}
\underbrace{\begin{bmatrix}
S_x & 0 & 0 \\
0 & S_y & 0 \\
0 & 0 & S_z
\end{bmatrix}}_{\text{Scaling } S}$$

Therefore:

$$
M = \begin{bmatrix}
R_{11} S_x & R_{12} S_y & R_{13} S_z \\
R_{21} S_x & R_{22} S_y & R_{23} S_z \\
R_{31} S_x & R_{32} S_y & R_{33} S_z
\end{bmatrix}
$$

As we can see $S_x$ only appears in the first column and have no effects on the second or the third column. 

For $j = k$ (e.g. how $S_x$ changes $R_{11} S_x$):

$$
\frac{\partial M_{ij}}{\partial S_k} = R_{ik}
$$

$\frac{\partial M_{ij}}{\partial S_k}$ means how much does value of Row i, Column j of M changes if I change $S_x$.

If $j = k$, $\frac{\partial M_{ij}}{\partial S_k}$ derivative is just the rotation value at the spot $R_{i,k}$. 

Otherwise the derivative $\frac{\partial M_{ij}}{\partial S_k}$ is 0 as we see in the example $S_x$ does nothing to different columns.

Therefore: 

$$\frac{\partial M_{i,j}}{\partial s_k} = \begin{cases} 
R_{i,k} & \text{if } j = k \\ 
0 & \text{otherwise} 
\end{cases}$$

Gradients for rotation $\frac{dM}{dq}$:

Recall quaternion is a way to represent 3D rotation using four numbers ($q_r, q_i, q_j, q_k$) instead of a 3x3 matrix. Its advantage is that it solves the Gimbal lock problem. Quaternions never lock up and allow Gaussians to rotate smoothly in any directions. Using quaternions let us nudge the four numbers slightly during optmization, as long as we normalise them, the result will always be a perfectly valid, clean rotation.

A unit quaternion q with real part $q_r$, and imaginary parts $q_i, q_j, q_k$ for a rotation matrix R {% cite rose2015quaternions %}:

$$R(q) = \begin{bmatrix}
1 - 2(q_j^2 + q_k^2) & 2(q_i q_j - q_r q_k) & 2(q_i q_k + q_r q_j) \\
2(q_i q_j + q_r q_k) & 1 - 2(q_i^2 + q_k^2) & 2(q_j q_k - q_r q_i) \\
2(q_i q_k - q_r q_j) & 2(q_j q_k + q_r q_i) & 1 - 2(q_i^2 + q_j^2)
\end{bmatrix}$$

- $q_r$ (real part) controls amount of rotation
- $q_i, q_j, q_k$ (imaginary parts) define the axis, the Gaussian is spinning around

Now we can find the gradients with different quoternion components:

Recall: 
$$
M = \begin{bmatrix}
R_{11} S_x & R_{12} S_y & R_{13} S_z \\
R_{21} S_x & R_{22} S_y & R_{23} S_z \\
R_{31} S_x & R_{32} S_y & R_{33} S_z
\end{bmatrix}
$$

$$R(q) = 2\begin{bmatrix}
(\frac{1}{2} - q_j^2 - q_k^2)S_x & (q_i q_j - q_r q_k)S_y & (q_i q_k + q_r q_j)S_z \\
(q_i q_j + q_r q_k)S_x & (\frac{1}{2} - q_i^2 - q_k^2)S_y & (q_j q_k - q_r q_i)S_z \\
(q_i q_k - q_r q_j)S_x & (q_j q_k + q_r q_i)S_y & (\frac{1}{2} - q_i^2 - q_j^2)S_z
\end{bmatrix}$$

Therefore we can find the derivatives of the 4 components of q:

$$
\begin{align*}
\frac{\partial M}{\partial q_r} &= 2 \begin{bmatrix} 0 & -s_y q_k & s_z q_j \\ s_x q_k & 0 & -s_z q_i \\ -s_x q_j & s_y q_i & 0 \end{bmatrix} \\[10pt]
\frac{\partial M}{\partial q_i} &= 2 \begin{bmatrix} 0 & s_y q_j & s_z q_k \\ s_x q_j & -2s_y q_i & -s_z q_r \\ s_x q_k & s_y q_r & -2s_z q_i \end{bmatrix} \\[10pt]
\frac{\partial M}{\partial q_j} &= 2 \begin{bmatrix} -2s_x q_j & s_y q_i & s_z q_r \\ s_x q_i & 0 & s_z q_k \\ -s_x q_r & s_y q_k & -2s_z q_j \end{bmatrix} \\[10pt]
\frac{\partial M}{\partial q_k} &= 2 \begin{bmatrix} -2s_x q_k & -s_y q_r & s_z q_i \\ s_x q_r & -2s_y q_k & s_z q_j \\ s_x q_i & s_y q_j & 0 \end{bmatrix}
\end{align*}
$$

Normlization of quoternions:

During forward pass, we perform a L2 normalization to the 4D quaternion vector q to ensure the rotation is "unit length":

$$
\hat{q} = \frac{q}{\|q\|}
$$

where 

$$
\|q\| = \sqrt{q_r^2 + q_i^2 + q_j^2 + q_k^2}
$$

Backward pass normalization ensures the gradient only updates the direction of the rotation, not the length of the vector:

$$\frac{\partial \mathcal{L}}{\partial q} = \frac{1}{\|q\|} \left( I - \hat{q}\hat{q}^T \right) \frac{\partial \mathcal{L}}{\partial \hat{q}}$$

- $\frac{\partial \mathcal{L}}{\partial q}$ represents the gradient of the loss function with respect to the raw quaternion parameters.

Kerbl uses Stochastic Gradient descent technique for the optimzation process, where the Gaussian splatting backpropagation involves these parameters;
- Positions(x, y, z)
- Opacity($\alpha$)
- Scaling(S)
- Rotation(q)
- Colors(SH)

The loss function $L_1$ is given:

$$ 
L = (1-\lambda)L_1 + \lambda L_{D-SSIM}
$$

- where SSIM stands for Structural Similarity Index, which mimics how humans see, emphasing more on structure, contrast and texture instead of perfect pixels. In all of Kerbl's tests $\lambda$ is set as 0.2
- $L_1$ is the absolute difference between color of pixel in the rendered image and the pixel of the original photo

For the learning rate for the training in Gaussians' positions, Exponential Decay is used to move them into their final, perfect position. Starting with a large learning rate to help the system converge, and the step size gradually shrinks to ensure the Gaussians make tiny position adjustments into their final pixel postiions

## Adative Density Control

Starting with an initial sparse scene from SfM, the goal is to grow into denser Gaussian sets to better represent the scene. Kerbl used a method called Adaptive Density Control interleaved during the optimization process.

Kerbl first performed a 100-iteration warm up period, then it performs a clean up where Gaussians with $\alpha$ less than the threshold $\epsilon_ \alpha$ get removed (as they are essentially transparent). This culling process helps to keep the scene efficient, and improve computations so that the GPU doesn't have to spend resources calculating its position, shape and sorting.

ADC will focus on regions with missing geometric features (under-reconstruction), but also regions where Gaussians cover large areas in the scene (over-reconstruction). Kerbl's paper observes that both of these regions have large view-space positional gradients, where in under-reconstruction region nearby Gaussian is desperately being pulled into the empty void to fill the gap. While in over-reconstruction region, the giant Gaussian is being moved back and forth aggressively to try make the image looks sharp.

In Kerbl's tests, they decided to densify Gaussians with average magnitude of view-space position gradients above the threshold $\tau_{pos}$ of 0.0002. For under-reconstructed regions, they simply clone the small Gaussians of the same size, and moving them towards the direction of positional  gradient. For over-reconstructed regions, giant Gaussians are replaced by smaller Gaussians, dividing their scale by factor of $\phi = 1.6$ (Kerbl determined experimentally).

To remove floaters close to the cameras, we reset the $\alpha$ values of all Gaussians to almost zero every N=3000 iterations. This tricks the loss function to help removing unneeded Gaussians, as the loss function will increase $\alpha$ aggressively for objects' Gaussians that are needed, while Gaussians with $\alpha$ less than $\epsilon_ \alpha$ will be pruned repeatedly. 

## Tile-based rasterizer

One of the reason why Kerbl's 3D Gaussian splatting is so fast compare to previous methods, is the fast differentiable tile-based rasterizer they implemented, which allows fast rednering and sorting to approximate $\alpha$-blending. This was inspired by previous work [Lassner and Zollhofer 2021].

This a fully differentiable alpha-blending pass, starting off by splitting the screen into 16x16 pixel tiles, it checks if a 3D Gaussian is inside the view frustum (camera's field of view). A list of splats are created per tile by instantiating each splat in each 16x16 tile it overlaps, which will first cause a moderate increase in the number of Gaussians.

$\underline{\textbf{Radix Sort}}$

Next we assign key for each splat's instance with up to 64 bits, where the higher 32 bits encode the index of the overlapped tile - the tile index. The lower 32 bits encode the projected depth. Note that if a Gaussian overlaps multiple tiles, it will recorded as multiple instances and each instance recieve a unique key. The sorting process will first organize which tile/tiles is each Gaussian in then it looks at the lower bits to sort each Gaussian in their respective tile from cloest to the camera to further away. This is called a Radix sort, which is much more efficient for CUDA programming as it's complexity becomes $O(n)$ instead of $O(nlogn)$ like QuickSort. The exact size of the index depends on the resolution.

Early exit can be applied from the results of Radix sort giving a front-to-back ordering. As we rasterize a pixel, we keep track of the accumulated transmittance $T$:

$$
C = \sum_{i=1}^{n} c_i \alpha_i T_i \quad \text{where} \quad T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)
$$

where:

- $C$ is the final pixel color after blending all the overlapping 3D Gaussians
- $c_i$ is the color of the $i$-th Gaussian in the depth ordering, derived from Spherical Harmonics
- $\alpha$ is the opacity of the $i$-th Gaussian
- $T_i$ is the Transmittance, the "visibility" remaining for the $i$-th Gaussian. How much light can still be pass through to reach this pixel

By sorting Gaussians in $\alpha$ and keeping track of accumulated transmittance, the renderer can stop processing further Gaussians when the accumulated opacity reaches 99.9%. This significally reduces the number of calculations, especially in high-depth scenes.

After sorting, we identify the start and end ranges in the sorted array with the same tile ID. Done in parallel, launching one thread per 64-bit array element comparing the higher 32 bits which is the tile ID with its neighbors. Compare to previous methods, Kerbl's method removes sequential primitive processing, every Gaussian is treated as an independent piece of data. They are also "move compact per-tile lists", meaning for each tile it only contains the Gaussians that actually contributed to those pixels.


This is the end of the article, hopefully it helps anyone who is also studying in this field ^^

A special thanks to the amazing youtube channel - Papers in 100 Lines of Code (https://www.youtube.com/@papersin100linesofcode/videos) with his amazing in depth videos in computer vision and neural networks.


## Citation
> **Sau Chi Tang.** (2026). *3D Gaussian Splatting Maths and Theory*. Sau Chi's Blog. https://sauchix.github.io/projects/gaussian-splatting/


## References

{% bibliography --cited %}

