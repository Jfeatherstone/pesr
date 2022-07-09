# <ins>P</ins>hoto<ins>e</ins>lastic <ins>S</ins>uper<ins>r</ins>esolution

Implemention of super-resolution algorithm described by [Shi et al. (2016)](https://arxiv.org/abs/1609.05158) trained specifically for photoelastic patterns in circular particles. Made to be used with force inversion algorithms like [pepe](https://github.com/Jfeatherstone/pepe).

![preliminary test on real data](https://raw.githubusercontent.com/Jfeatherstone/pesr/master/prelim_real_test.png)

Above result is for a very low resolution image (particle is 15 pixels across) taken from a real photoelastic system; network had been trained on about 5 million synthetic images here (no real images though). Generally, for the force inversion process to work, you want at least 50 pixels across a particle, so this may allow force inversion on data for which high resolution imaging is not possible. Force inversion fails entirely if run on the raw input image; high resolution reference is shown on the right, but this is not used in the inversion process at all.
