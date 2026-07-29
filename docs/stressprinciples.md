# Stress Principles

While not as fully developed as the strain-rate options, `strain_tools` does provide methods to convert strain-rate values into deviatoric and Cauchy stresses.

## Deviatoric Stress

We can convert from surface strain rates to deviatoric surface stresses using the generalised isotropic form of Glen's Flow Law:

\begin{equation}
\tau_{ij} = A^{-\frac{1}{n}} \dot{\varepsilon}_E^{\frac{1-n}{n}} \dot{\varepsilon}_{ij}
\end{equation}

Here, $\dot{\varepsilon}_{ij}$ is the given component of the strain-rate tensor, $\dot{\varepsilon}_{E}$ is the effective strain rate, $n$ is the creep exponent, and $A$ is the temperature-dependent creep-rate factor in the form

\begin{equation}
A = A_* \exp \left[ -\frac{Q_c}{R} \left( \frac{1}{T_h} - \frac{1}{T_*} \right) \right]
\end{equation}

We use Cuffey and Paterson (2010)'s recommended values, as commonly used across the literature:

\begin{align*}
A_* &= 3.5 \times 10^{-25} \quad \mathrm{Pa^{-3}\,s^{-1}} \\
n   &= 3 \\ 
T_* &= 263 + 7 \times 10^{-8}P \quad \mathrm{K} \\
T_h &= T + 7 \times 10^{-8}P \quad \mathrm{K} \\
Q_c &= 
\begin{cases} 
Q^- = 60 \quad \mathrm{kJ\,mol^{-1}}, & \text{if } T_h \lt T_* \\[3pt]
Q^+ = 115 \quad \mathrm{kJ\,mol^{-1}}, & \text{if } T_h \geq T_* 
\end{cases}
\end{align*}


At the surface, pressure values are assumed to be zero ($P=0$) and thus temperature values simplify to $T_* = 263 \textnormal{ K}$ and $T_h = T$. Temperature can be provided as a single value (often chosen at -5˚C) or as a gridded set of values at the same resolution and extent as the velocity field (e.g. RACMO- or similar reanalysis-derived long-term surface temperature averages).

!!! question "No option for $n=4$?"

    Given recent discussion in the wider literature recommending $n=4$ as a more representative flow exponent, some studies have explored deriving stresses from horizontal velocity fields using $n=4$ (e.g. Wells-Moran _et al._ 2025; Reynolds _et al._ 2025). However, these approaches have yet to consolidate around a single accepted 'off-the-shelf' method that mirrors the Cuffey and Paterson (2010) values for $n=3$ (primarily due to uncertainty in $A_0$ or $A_*$ prefactor values). As such, the current `strain_tools` implementation does not provide an $n=4$ approach, but welcomes recommendations with the aim of including such capability in a future update.

    <!-- -
    For moving to an n=4 system - Wells-Moran and Reynolds papers both approach A using A_0 constant prefactor
    $$
    A = A_0 \exp \left( \frac{-Q_c}{RT} \right)
    $$ 
    -->

## Cauchy Stress

We can calculate the surface-parallel Cauchy stress tensor from a given deviatoric stress tensor using the isotropic pressure ($p$):

\begin{equation}
\sigma_{ij} = \tau_{ij} + p \delta_{ij},
\end{equation}

where $p = \frac{1}{3} (\sigma_{xx} + \sigma_{yy} + \sigma_{zz})$ and $\delta_{ij}$ denotes the Kronecker delta ($\delta_{ij}=1$ if $i=j$ and $\delta_{ij}=0$ if $i \neq j$). 

Under the assumption of zero vertical normal stress at the surface ($\sigma_{zz} = 0$), the isotropic pressure $p$ simplifies to

\begin{equation}
p = \tau_{xx} + \tau_{yy} = \tau_{1} + \tau_{2}.
\end{equation}

Hence, for any __normal stress component__ (where $i=j$: e.g. principal, longitudinal, transverse stress),

\begin{equation}
\sigma_{ij} = \tau_{ij} + \tau_{xx} + \tau_{yy} = \tau_{ij} + \tau_{1} + \tau_{2}.
\end{equation}

Note that because the isotropic pressure $p \delta_{ij}$ acts uniformly in all directions, adding $p$ shifts the stress magnitudes but does not alter the stress orientations.

For any __shear stress component__ (where $i \neq j$), $p \delta_{ij}$ resolves to zero and thus no conversion needs to be performed:

\begin{equation}
\sigma_{ij} = \tau_{ij},
\end{equation}


<!-- ## More stresses - Von Mises, Schmid-Ishlinsky, Mohr-Coulomb, Drucker–Prager, Hayhurst?

All summarised by Culberg 2026 / Wells-Moran 2025 -->