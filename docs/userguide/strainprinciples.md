# Strain Rate Principles

## Calculating Strain Rate

We define the surface strain rate tensor $\epsilon_{ij}$ in terms of the surface-parallel components of velocity $u$ and $v$ in projection coordinate directions $x$ and $y$ as

$$
\dot{\varepsilon}_{ij}
=
\begin{bmatrix}
\dfrac{\partial u}{\partial x}
&
\dfrac{1}{2}\left(\dfrac{\partial v}{\partial x} + \dfrac{\partial u}{\partial y}\right)
\\[8pt]
\dfrac{1}{2}\left(\dfrac{\partial v}{\partial x} + \dfrac{\partial u}{\partial y}\right)
&
\dfrac{\partial v}{\partial y}
\end{bmatrix}
=
\begin{bmatrix}
\dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{xy} & \dot{\varepsilon}_{yy}
\end{bmatrix}.
$$

Strain rates are defined as positive in extension and negative in compression. 

### Nominal Strain Rate

Nominal strain rates are calculated using the finite difference of the velocity field: here, using `numpy.diff()`.

### Logarithmic Strain Rate

We calculate logarithmic strain rates following Alley _et al._ (2018). This approach calculates integrated deformation of ice parcels and is more resistant to errors than nominal strain rates in situations where large velocity gradients occur (such as at crevasse fields). The method requires a length scale value ($r$) to be determined as the radius from the pixel center over which strain is calculated.

## Strain Types

### Effective Strain Rate

$$ 
    \sqrt{ \frac{1}{2} \left( e_{xx}^2 + e_{yy}^2 + e_{xy}^2 \right) }, 
$$

following Cuffey & Paterson (2010, p.59). 

### Principal Strain Rates

#### Eigenvectors

#### Eigenvalues

Following Nye (1959) and Harper et al. (1998) and returned as e_1 and e_2 This is quicker to compute, but only returns magnitude values.

### Longitudinal, Transverse, and Shear Strain Rates

Following _Bindschadler et al._ (1996).

### Strain rate uncertainty

Calculate strain rate uncertainty following Poinar and Andrews (2021, eq. 4):

$$
    \delta_{\dot{\epsilon}} = \frac{1}{\Delta x} \sqrt{(\delta u)^2 + (\delta v)^2}
$$

Where Δx is the baseline distance between observation points (i.e. the lengthscale), and δu and δv are the velocity uncertainties in the x and y directions.