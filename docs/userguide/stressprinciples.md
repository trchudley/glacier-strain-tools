# Stress

## Deviatoric Stress

Calculate deviatoric stress for a given strain rate field.

Strain rates assumed to be in units of a-1 unless otherwise specified.

Parameters following Wells-Moran et al. (2024).

## Cauchy Stress

Calculate Cauchy stress for a given deviatoric stress using the first and second principal deviatoric stresses. These are related through the isotropic pressure such that:

$$ \sigma_{ij} = \tau_{ij} + p \delta_{ij}, $$

where $p = \frac{1}{3} (\sigma_{1} + \sigma_{2} + \sigma_{zz})$. Assuming that $\sigma_{zz} = 0$, $p = \frac{1}{3} (\sigma_{1} + \sigma_{2})$ which $= \tau{1} + \tau{2}$. Hence,

$$ \sigma_{ij} = \tau_{ij} + \tau_{1} + \tau_{2}. $$

## Von Mises

## Mohr-Coulomb, Drucker–Prager, Hayhurst

Wells-Moran

## 