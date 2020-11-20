from pomegranate import (
    NaiveBayes,
    NormalDistribution,
    UniformDistribution,
    ExponentialDistribution,
    GeneralMixtureModel,
    MultivariateGaussianDistribution,
    BernoulliDistribution,
)
import pandas as pd
import numpy as np

X = pd.DataFrame({"A": [1, 0, 1, 0, 1], "B": [1, 1, 1, 1, 0]})

x = BernoulliDistribution(0.4)

vals = []
[vals.append(x.sample()) for i in range(1000)]

model = NaiveBayes(
    [NormalDistribution(5, 2), UniformDistribution(0, 10), ExponentialDistribution(1.0)]
)
model.predict(np.array([[10]]))

model = GeneralMixtureModel.from_samples(
    MultivariateGaussianDistribution, n_components=3, X=X
)
