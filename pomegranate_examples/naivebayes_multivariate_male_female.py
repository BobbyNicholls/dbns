#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Male or Female Multivariate

# author: Nicholas Farn [<a href="sendto:nicholasfarn@gmail.com">nicholasfarn@gmail.com</a>]

# This example shows how to create a Multivariate Guassian Naive Bayes Classifier using pomegranate. In this example we will use a set od data measuring a person's height (feet), weight (lbs), and foot size (inches) in order to classify them as male or female. This example is drawn from the example in the Wikipedia <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Examples">article</a> on Naive Bayes Classifiers.

# In[1]:


from pomegranate import *
import numpy as np


# Since we are simply using two Multivariate Gaussian Distributions, our Naive Bayes model is very simple to initialize.

# In[2]:


model = NaiveBayes(
    [
        MultivariateGaussianDistribution,
        MultivariateGaussianDistribution,
        MultivariateGaussianDistribution,
        MultivariateGaussianDistribution,
    ]
)


# Of course currently our model is unitialized and needs data in order to be able to classify people as male or female. So let's create the data. For multivariate distributions, the training data set has to be specified as a list of lists with each entry being a single case for the data set. We will specify males as a 0 and females with a 1.

# In[3]:


X = np.array(
    [
        [6, 180, 12],
        [5.92, 190, 11],
        [5.58, 170, 12],
        [5.92, 165, 10],
        [6, 160, 9],
        [5, 100, 6],
        [5.5, 100, 8],
        [5.42, 130, 7],
        [5.75, 150, 9],
        [5.5, 140, 8],
    ]
)

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


# Now we can fit our Naive Bayes model to the set of data.

# In[4]:


model.fit(X, y)


# Now let's test our model on the following sample.

# In[5]:


data = np.array([[5.75, 130, 8]])


# First the probability of the data occurring under each model.

# In[6]:


for sample, probs in zip(data, model.predict_proba(data)):
    print "Height {}, weight {}, and foot size {} is {:.3}% male, {:.3}% female.".format(
        sample[0], sample[1], sample[2], 100 * probs[0], 100 * probs[1]
    )


# We can see that the probability that the sample is a female is significantly larger than the probability that it is male. Logically when we classify the data as either male (0) or female (1) we get the output: female.

# In[7]:


for sample, result in zip(data, model.predict(data)):
    print "Person with height {}, weight {}, and foot size {} is {}".format(
        sample[0], sample[1], sample[2], "female" if result else "male"
    )
