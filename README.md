# Uncertainty Quantifications in Spiking Neural Networks 


### Probabalistic and Bayesian Learning: A look on traditional learning methods
**Probabilistic Learning** is the most common form of learning in the field of Machine Learning. A model is shown data and, sometimes, a label and returns a prediction, varying to the task at hand. These predictions are point estimates, elusive in the full spread of predictions or variance of the predicted target. More often than not, a dataset can be mathematically represented by a large set of functions, instead of just a singular.

**Bayesian Learning** aims to give a deeper sense of the model's predictions by representing the model's parameters as a probability distribution function. True to its name, Bayes' Rule allows a model to continously update its distribution function of parameters during learning, as it observes new data and compares the new data on a hypothesis. 

$$ P(Hypothesis _given_ Evidence) =  {P(Evidence _given_ Hypothesis) + P(Hypothesis)} {\over P(Evidence)}$$

In Bayesian Inference: 
- We start with an initial belief - a random, normal distibution of weights of our parameters that may fit our data. There are multiple initial beliefs that perform the best.

- We observe new data and evaluate the probability of our hypothesis occuring with the new data. 

- We update our initial belief with the new information and repeat the observation step, this time with new data.

### Important Terms to consider:

- **Likelihood** is the probability of observing the data given our hypothesis is true. The likelihood quantifies how well the weight explains the training data. 
- **Marginal Likelihood** is the probability of observing the data under _all_ possible assumptions. This is the evidence, or the probability of observing the training data per network weight. 
- **Posterior** represents an updated belief of the hypothesis after observing the new data. The goal during Bayesian Inference is the find a posterior distribution that maximizes the log marginal likelihood.
- **Confidence** is not the same as likelihood. Consider a classification model trained with a softmax activation function. Even if we feed in out-of-sample data to our trained classification model, the model *must* return probabilities summed to one. These output probabilities are severely unreliable. There is no uncertainty about the data represented by the model in these prediction scores - the model is unreliably confident. 
- **Variance** is one of the most important terms in reliability evaluation. On a higher level, variance can be used to quantify the variability or inconsistency in predictions made. Ensembles can be used to measure variance between predictions to evaluate how similar models vary in their predictions and weight parameters. Variance can also be used to evaluate variability in datasets. 

### Uncertainty in DL:
We are really just trying to measure how reliable a model is. Two types of uncertainties help us with that. 

- **Epistemic Uncertainty** is the uncertainty inherent in the model, emphasis on the systemic part of epistemic. A model can be very confident in its prediction but still have high epistemic uncertainty. This value is high when the training data is incomplete or poorly representative of the data used to evaluate the model. Epistemic uncertainty can be reduced by adding more data so the model learns the out-of-sample patterns.

- **Aleatoric uncertainty** is the inherent noise in data collection. This uncertainty is irreducible outside of data transformation. It is high when the input data is noisy (which can arise from a variety of factors) and cannot be reduced by adding more data. 

A traditional Bayesian Neural Network (BNNs) will represent weights over a distribution and likelihood function. These Bayesian nets learn a posterior distribution over weights using Bayes Rule - and since these distributions are intractable to compute (we must evaluate probility of all possible values, really computationaly expensive), we must sample to approximate the posterior. There are a couple solutions to this: 
- **Dropout** during testing time implies a Bernoulli random variable with probability p to randomly drop neurons during training. This reduces overfitting. After the input is passed into the model for testing, we measure samples.
- **Model ensembles** represent a collection of networks. Each network has a unique set of weights it has learned after it has seen a unique set of training data (often times, the data is the same, but in differing orders).

Both of these techniques produce a variance and expected value. If none of the outputs agree with each other (if the variance is unreasonably high), then we have high epistemic uncertainty.


### Why Bayesian Learning? 
We can measure uncertainty inherent in a model when exposed to some data. We may be able to represent reasons why a model makes classifications based on probabilities instead of input features. 

### Why not Bayesian Learning?
Expensive. Very. A Monte Carlo approach will require massive amounts of simulation. Furthermore, we are running the network several times for every input. 

Ensembles require multiple models to train and simulate; averaging weigthts every time. This can lead to memory constraints, even more so on edge devices. 

### Uncertainty can stem from:
a. Data measurements
b. A confident, but unaccurate model's predictions

Problem b can be solved quite intuitively. If a model with high accuracy consistently predicts a certain type of data incorrectly, we can feed it similar data and have it train on that until it perfects it. With enough epochs, a model can quickly converge to a new set of parameters that fit the new data (this is true in both Bayesian and Frequentist Learning). As variance in data grows, we need larger samples to have the same amount of confidence as smaller ones.

Problem a can also be solved intuitively, but not with adjustments to model or training techniques. It is also difficult to measure what is out-of-sample and what is not. 


### Sources: 
    arXiv:2401.07359 [cs.AI]



