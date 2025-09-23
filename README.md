# studenttmixture

Mixtures of multivariate Student's t distributions are widely used for clustering
data that may contain outliers, but scipy and scikit-learn do not at present
offer classes for fitting Student's t mixture models. This package provides classes
for:

1) Modeling / clustering a dataset using a finite mixture of multivariate Student's
t distributions fit via the EM algorithm. This is analogous to scikit-learn's 
GaussianMixture.
2) Modeling / clustering a dataset using a mixture of multivariate Student's 
t distributions fit via the variational mean-field approximation. This is analogous to
scikit-learn's BayesianGaussianMixture.

### Installation

    pip install studenttmixture

Starting with version 1.11, this is a pure Python package so installation
should be very straightforward.

Dependencies are numpy, scipy and scikit-learn.

### Usage

- [EMStudentMixture](https://github.com/jlparkI/mix_T/blob/main/docs/Finite_Mixture_Docs.md)<br>
- [VariationalStudentMixture](https://github.com/jlparkI/mix_T/blob/main/docs/Variational_Mixture_Docs.md)<br>
- [Tutorial: Modeling with mixtures](https://github.com/jlparkI/mix_T/blob/main/docs/Tutorial.md)<br>

### Background

- [Deriving the mean-field formula](https://github.com/jlparkI/mix_T/blob/main/docs/variational_mean_field.pdf)<br>
