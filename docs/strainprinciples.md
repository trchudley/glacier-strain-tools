# Strain Rate Principles

## Calculating Strain Rate

!!! info
    Methods to calculate logarithmic strain rates from velocity fields are `numba`-accelerated Python adaptations of the MATLAB methods presented by Alley _et al._ (2018), and the methods presented here summarise that of the associated paper closely. For full derivations, we recommend consulting the methods and supplementary material of Alley _et al._ (2018). 


We define the 2-D surface strain rate tensor $\dot{\varepsilon}_{ij}$ in terms of the surface-parallel components of velocity $u$ and $v$ in projection coordinate directions $x$ and $y$ as

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

!!! note
    When working with remotely sensed ice velocity data $u$ and $v$ are more commonly referred to as $v_x$ and $v_y$. For consistency, this theory page will refer to $u$ and $v$ throughout - but within the explanatory notebooks and API, when real data is being used, $v_x$ and $v_y$ will be the preferred terms.

### Nominal Strain Rate

The simplest formulation for deriving strain rates from satellite-derived velocity fields is known as 'nominal' or 'engineering' strain, defined as the change in length of a parcel ($\delta L$) over the original length ($L_0$) per unit time ($\delta t$).

\begin{equation}
\dot{\varepsilon} = \frac{\delta L / L_0}{\delta t}
\end{equation}

In practice, this is approximated by differencing velocities over a finite distance and dividing by the offset distance, for instance:

\begin{equation}
\frac{\textnormal{d} u}{\textnormal{d} x} = \frac{u_2 - u_1}{\Delta x}
\end{equation}

In programming terms, this can be achieved by calculating finite difference of the velocity field: here, using `numpy.diff()`.


### Logarithmic Strain Rate

The nominal strain rate is derived under the assumption that $\delta L$ is very small compared to $L_0$, and the practical implementation assumes that velocities change linearly between sample points. These assumptions begin to break down when a parcel is strained significantly (e.g. at shear margins). An alternative, albeit slightly more complex to derive, definition compares the length change with a previous length, rather than original length. This quantity is known as the 'logarithmic' or 'true' strain, and was described by Nye (1959) as

\begin{equation}
\dot{\varepsilon} = \frac{1}{\Delta t} \ln\left(\frac{L_f}{L_0}\right),
\end{equation}

where $L_f$ is the final length of the parcel.

Nye's approach was to measure stake movement around a grid of five stakes, organised in a 'diamond' pattern around a central point. Logarithmic strain rates could be measured for each segment and averaged to measure four strain rates: $a$, $b$, $c$, and $d$ (at 0˚, 45˚, 90˚, and 135˚ angles respectively), from which the strain rate tensor could be calculated:

<figure markdown="span">
  ![Visualisation of strain grid](../assets/nye_strain_alley_2018.gif){ width="80%" }
  <figcaption>Visualisation of Nye's strain grid, from Alley <i>et al.</i> (2018; Figure 1)</figcaption>
</figure>

\begin{equation}
\dot{\varepsilon}_{xx} = -\frac{1}{4}\dot{\varepsilon}_0 + \frac{1}{4}\dot{\varepsilon}_{45} + \frac{3}{4}\dot{\varepsilon}_{90} + \frac{1}{4}\dot{\varepsilon}_{135}
\end{equation}

\begin{equation}
\dot{\varepsilon}_{xy} = \frac{1}{2}\dot{\varepsilon}_{45} - \frac{1}{2}\dot{\varepsilon}_{135}
\end{equation}

\begin{equation}
\dot{\varepsilon}_{yy} = \frac{3}{4}\dot{\varepsilon}_0 + \frac{1}{4}\dot{\varepsilon}_{45} - \frac{1}{4}\dot{\varepsilon}_{90} + \frac{1}{4}\dot{\varepsilon}_{135}
\end{equation}

The approach of Alley _et al._ (2018) virtualises this method, using a digital 'stake' placed at the centre of the grid with four more placed a given length scale ($r$) away from the central stake. Stakes are allowed to advect through the velocity field before logarithmic strain rates are calculated.

This approach can be more computationally intensive than nominal strain rates, although the use of `numba`-wrapped functions within `strain_tools` aims to ameliorate this issue. In return, this method offers several advantages:

 - Velocity is sampled continuously along the advected stake trajectories, which means (unlike the nominal strain rate method) that nonlinear velocity variations between sample points can be identified.
 - As a rule of thumb, nominal and logarithmic strain will be approximately at $\delta L / L_0$ values of <2%. Beyond this, logarithmic strain rates remain a more accurate approximation of the real strain rate under finite deformation, and as such is more resistant to errors than nominal strain rates in situations where large velocity gradients occur (such as at crevasse fields). 
 - The method accepts a length scale value ($r$) to be determined as the radius from the pixel center over which strain is calculated: the nominal strain rate code, as provided, uses `numpy.diff()`, and as such the strain rate is always calculated at the resolution of the velocity grid. (Note that you could use filtering to calculate the gradients over larger windows - for instance, Minchew _et al._ (2017) use a second-order Savitsky–Golay filter to calculate gradients over 2 km windows).

The logarithmic strain rate is the recommended (and default) approach within `strain_tools`, unless processing constraints are such that nominal strain rates become a drastically more efficient option.

## Derivative Strain Rates

### Principal Strain Rates

Within `strain_tools`, the first and second principal strain rates (\(\dot{\varepsilon}_1\) and \(\dot{\varepsilon}_2\)) are defined as the maximum (most extensional) and minimum (most compressive) surface-parallel strain rates. Note this is one of two conventions within glaciology: the other refers to these as \(\dot{\varepsilon}_1\) and \(\dot{\varepsilon}_3\), with \(\dot{\varepsilon}_2\) being the strain rate in the vertical axis.

#### Eigenvectors

The principal strain rates and their vectors can be calculated directly by calculating the eigenvalues of the surface-parallel strain-rate tensor:

\begin{equation}
\dot{\varepsilon}_{ij} \textbf{v}_i = \dot{\varepsilon}_i \textbf{v}_i
\end{equation}

where \(\textbf{v}_i\) are the eigenvectors associated with the principal strain rates \(\dot{\varepsilon}_i\).

In practice, this is simply done using the `numpy.linalg.eigh()` function.

<!-- 
The orientation of each principal strain rate is obtained from the corresponding eigenvector:

\begin{equation}
\theta_i = \tan^{-1}\left(\frac{v_{iy}}{v_{ix}}\right)
\end{equation}

where \(v_{ix}\) and \(v_{iy}\) are the x and y components of the eigenvector associated with \(\dot{\varepsilon}_i\). 
-->

#### Eigenvalues

Alternatively we can follow Nye (1959) to calculate the eigenvalues. This is slightly more computationally efficient, but does not return the vector components (so cannot be used for plotting, etc.)

\begin{equation}
\dot{\varepsilon}_1, \dot{\varepsilon}_2 = \frac{1}{2} (\dot{\varepsilon}_{xx} + \dot{\varepsilon}_{yy}) \pm \sqrt{ \frac{1}{4} (\dot{\varepsilon}_{xx} - \dot{\varepsilon}_{yy})^2 + \dot{\varepsilon}_{xy}^2 }
\end{equation}

<!-- 
The orientation of the maximum principal stress relative to the $x$-axis can be calculated as follows:

\begin{equation}
\theta_1 = \frac{1}{2} \tan^{-1} \left( \frac{2 \dot{\varepsilon}_{xy}}{\dot{\varepsilon}_{xx} - \dot{\varepsilon}_{yy}} \right)
\end{equation}

The orientation of the minimum principal strain rate is $\theta_1$ rotated by $90^\circ$ ($\pi/2$ radians):

\begin{equation}
\theta_2 = \theta_1 + \frac{\pi}{2}
\end{equation} 

From the eigenvalues and the orientations, you could then calculate the eigenvectors as follows:

\begin{equation}
\textbf{v}_1
\end{equation}

-->

### Longitudinal, Transverse, and Shear Strain Rates

Rotating the strain rates to orient the components relative to the local flow direction can be calculated, given the local flow direction $\theta$ measured anti-clockwise from the positive $x$ axis:

\begin{equation}
\theta = \tan^{-1}\left( \frac{v}{u} \right)
\end{equation}

The longitudinal ($\dot{\varepsilon}_{lon}$, along-flow), transverse ($\dot{\varepsilon}_{trn}$ perpendicular to flow), and shear ($\dot{\varepsilon}_{shr}$) strain rates can be calculated following Bindschadler _et al._ (1996) as follows:

\begin{equation}
\dot{\varepsilon}_{lon} = \dot{\varepsilon}_{xx} \cos^2 \theta + 2 \dot{\varepsilon}_{xy} \cos \theta \sin \theta + \dot{\varepsilon}_{yy} \sin^2 \theta
\end{equation}

\begin{equation}
\dot{\varepsilon}_{trn} = \dot{\varepsilon}_{xx} \sin^2 \theta - 2 \dot{\varepsilon}_{xy} \cos \theta \sin \theta + \dot{\varepsilon}_{yy} \cos^2 \theta
\end{equation}

\begin{equation}
\dot{\varepsilon}_{shr} = (\dot{\varepsilon}_{yy} - \dot{\varepsilon}_{xx}) \cos \theta \sin \theta + \dot{\varepsilon}_{xy} (\cos^2 \theta - \sin^2 \theta)
\end{equation}

### Effective Strain Rate

The full, three-dimensional effective strain rate is defined as

\begin{equation}
\dot{\varepsilon}_{E} = \sqrt{ \frac{1}{2} ( \dot{\varepsilon}_{xx}^2 + \dot{\varepsilon}_{yy}^2 + \dot{\varepsilon}_{zz}^2 ) + \dot{\varepsilon}_{xy}^2 + \dot{\varepsilon}_{xz}^2 + \dot{\varepsilon}_{yz}^2 },
\end{equation}

given that $\dot{\varepsilon}_{xz}$ and $\dot{\varepsilon}_{yz}$ will be zero, and, assuming incompressibility, $\dot{\varepsilon}_{zz}$ can be inferred from $(-\dot{\varepsilon}_{xx} - \dot{\varepsilon}_{yy})$, we calculate effective strain rate as

\begin{equation}
\dot{\varepsilon}_{E} = \sqrt{ \frac{1}{2} [ \dot{\varepsilon}_{xx}^2 + \dot{\varepsilon}_{yy}^2 + (-\dot{\varepsilon}_{xx} - \dot{\varepsilon}_{yy})^2 ] + \dot{\varepsilon}_{xy}^2 }.
\end{equation}

Some alternative implementations neglect the vertical strain rate (assuming it to be 0 at all time), which gives a _planar_ effective strain rate as

\begin{equation}
\dot{\varepsilon}_{E} = \sqrt{ \frac{1}{2} \left( \dot{\varepsilon}_{xx}^2 + \dot{\varepsilon}_{yy}^2 \right)+ \dot{\varepsilon}_{xy}^2 }.
\end{equation}

This formulation ignores the vertical strain-rate component rather than enforcing incompressibility, and therefore represents a purely planar invariant. This could be seen as useful in problems dealing with crevasse opening/closing, but this was not found to be better when tested by Reynolds _et al._ (2025).

### Strain rate uncertainty

Strain rate uncertainty can be calculated following Poinar and Andrews (2021, eq. 4):

\begin{equation}
\delta_{\dot{\epsilon}} = \frac{1}{\Delta x} \sqrt{(\delta u)^2 + (\delta v)^2}
\end{equation}

Where $\Delta x$ is the baseline distance between observation points (i.e. the length scale), and $\delta u$ and $\delta v$ are the velocity uncertainties in the $x$ and $y$ directions.