from pomegranate import (
    NaiveBayes,
    NormalDistribution,
    UniformDistribution,
    ExponentialDistribution,
    GeneralMixtureModel,
    MultivariateGaussianDistribution,
)
import pandas as pd
import numpy as np

X = pd.DataFrame({
    'A' : [1,2,3,4,5],
    'B' : [1,2,3,4,5]
})

model = NaiveBayes([NormalDistribution(5, 2), UniformDistribution(0, 10), ExponentialDistribution(1.0)])
model.predict(np.array([[10]]))

model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=3, X=X)

