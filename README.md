What is Machine Learning? Well, Machine Learning is a concept that allows the machine to learn from examples and experience, and that too without being explicitly programmed. So instead of you writing the code, what you do is feed data to the generic algorithm, and the algorithm machine builds the logic based on the given data.

This blog on Machine learning Algorithms will make your understanding more clear and will set a foundation for Machine Learning Certification Training. This blog will tell you about:

What is Machine Learning?
How does Machine Learning work?
Type Of Problems In Machine Learning 
Machine Learning Types
Machine Learning Algorithms
What is Machine Learning?

The term Machine Learning was first coined by Arthur Samuel in the year 1959. Looking back, that year was probably the most significant in terms of technological advancements.

If you browse through the net about ‘what is Machine Learning, you’ll get at least 100 different definitions. However, the very first formal definition was given by Tom M. Mitchell:

[fancyquote]

A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.

[/fancyquote]




In simple terms, Machine learning is a subset of Artificial Intelligence (AI) which provides machines the ability to learn automatically & improve from experience without being explicitly programmed to do so. In this sense, it is the practice of getting Machines to solve problems by gaining the ability to think.

But wait, can a machine think or make decisions? Well, if you feed a machine a good amount of data, it will learn how to interpret, process, and analyze this data by using Machine Learning Algorithms, in order to solve real-world problems. 

Before moving any further, let’s discuss How Machine Learning Works.

How does Machine Learning Work?

The Machine Learning algorithm is trained using a training data set to create a model. When new input data is introduced to the ML algorithm, it makes a prediction on the basis of the model.

The prediction is evaluated for accuracy and if the accuracy is acceptable, the Machine Learning algorithm is deployed 

This is just a very high-level example as there are many factors and other steps involved.

Types of Problems in Machine Learning

There are three main types of problems:

Regression: In this type of problem the output is a continuous quantity. So, for example, if you want to predict the speed of a car given the distance, it is a Regression problem. We solve this case using Supervised Learning algorithms like Linear Regression.
Classification: In this type, the output is a categorical value. Classifying emails into two classes, spam and non-spam is a classification problem that can be solved by using Support Vector Machines, Naive Bayes, Logistic Regression, K Nearest Neighbor, etc.
Clustering: This type of problem involves assigning the input into two or more clusters based on feature similarity. For example, clustering viewers into similar groups based on their interests, age, geography, etc.

Now that you have a good idea about what Machine Learning is and the processes involved in it, let’s now see various types of Machine Learning.

Machine Learning Types

A machine can learn to solve a problem by following any one of the following three approaches. These are the ways in which a machine can learn:

Supervised Learning
Unsupervised Learning
Reinforcement Learning
Supervised Learning

Supervised learning is a technique in which we teach or train the machine using data that is well labeled.

To understand Supervised Learning let’s consider an analogy. As kids we all needed guidance to solve math problems. As our teachers helped us understand what addiction is and how it is done. Similarly, you can think of supervised learning as a type of Machine Learning that involves a guide. The labeled data set is the teacher that will train you to understand patterns in the data. The labeled data set is nothing but the training data set.

Consider the above figure. Here we’re feeding the machine images of Tom and Jerry and the goal is for the machine to identify and classify the images into two groups (Tom images and Jerry images). The Labeled Data is fed to model, as in, we’re telling the machine, ‘this is how Tom looks and this is Jerry’. By doing so you’re training the machine by using labeled data. In Supervised Learning, there is a well-defined training phase done with the help of labeled data.

Unsupervised Learning

Unsupervised learning involves training by using unlabeled data and allowing the model to act on that information without guidance.
Think of unsupervised learning as a smart kid that learns without any guidance. In this type of Machine Learning, the model is not fed with labeled data, as in the model has no clue that ‘this image is Tom and this is Jerry’, it figures out patterns and the differences between Tom and Jerry on its own by taking in tons of data.

For example, it identifies prominent features of Tom such as pointy ears, bigger size, etc, to understand that this image is of type 1. Similarly, it finds such features in Jerry and knows that this image is of type 2. Therefore, it classifies the images into two different classes without knowing who Tom is or Jerry is.

Reinforcement Learning

Reinforcement Learning is a part of Machine learning. The agent is in an environment. The agent learns to behave in this environment by performing certain actions and observing the rewards which it gets from those actions.

This type of Machine Learning is comparatively different. Imagine you were dropped off at an isolated island! What would you do?
Panic? Yes, of course, initially we all would. But as time passes by, you will learn how to live on the island. You will explore the environment, understand the climate condition, the type of food that grows there, the dangers of the island, etc. This is exactly how Reinforcement Learning works, it involves an Agent (you, stuck on the island) that is put in an unknown environment (island), where he must learn by observing and performing actions that result in rewards.

We use, Reinforcement Learning in advanced Machine Learning areas. i.e self-driving cars, AplhaGo, etc.

So that sums up the types of Machine Learning. Now, let’s look at the types of Machine Learning Algorithms.

Machine Learning Algorithms

The list of the 5 most commonly used machine learning algorithms.

Linear Regression
Logistic Regression
Decision Tree
Naive Bayes
CNN
1. Linear Regression

Linear regression is used to estimate real values based on continuous variables. Here, we establish a relationship between the independent and dependent variables by fitting the best line. This best fit line is known as the regression line and represented by a linear equation Y= aX + b.
The best way to understand linear regression is to relive this experience of childhood. Let us say, you ask a child in fifth grade to arrange people in his class by increasing order of weight, without asking them their weights! What do you think the child will do? He/she would likely look (visually analyze) at the height and build of people and arrange them using a combination of these visible parameters. This is linear regression in real life! The child has actually figured out that height and build would be correlated to the weight by a relationship, which looks like the equation above.

In this equation:

Y – Dependent Variable
a – Slope
X – Independent variable
b – Intercept

These coefficients a and b are derived based on minimizing the ‘sum of squared differences of the distance between data points and regression line.

[code]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# generate random data-set
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(x, y)
# Predict
y_predicted = regression_model.predict(x)

[/code]

Look at the plot given. Here, we have identified the best fit having linear equation  y=0.2811x+13.9.

Now using this equation, we can find the weight, knowing the height of a person.

2. Logistic regression

Don’t get confused by its name! It is a classification, and not a regression algorithm. It is used to estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on a given set of the independent variable(s). In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function. Hence, it is also known as logit regression. Since it predicts the probability, its output values lie between 0 and 1.

Again, let us try and understand this through a simple example.

Let’s say your friend gives you a puzzle to solve. There are only 2 outcome scenarios – either you solve it or you don’t. Now imagine, that you are being given a wide range of puzzles/quizzes in an attempt to understand which subjects you are good at. The outcome of this study would be something like this – if you are given a trigonometry-based tenth-grade problem, you are 70% likely to solve it. On the other hand, if it is a grade fifth history question, the probability of getting an answer is only 30%. This is what Logistic Regression provides you.

Coming to the math, the log odds of the outcome is modeled as a linear combination of the predictor variables.

Above, p is the probability of the presence of the characteristic of interest. It chooses parameters that maximize the likelihood of observing the sample values rather than that minimize the sum of squared errors (like in ordinary regression).

Now, you may ask, why take a log? For the sake of simplicity, let’s just say that this is one of the best mathematical ways to replicate a step function. I can go in more detail, but that will beat the purpose of this blog.

[code]
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear', random_state=0)

model.fit(x, y)

model.predict(x)

[/code]
There are many different steps that could be tried in order to improve the model:
including interaction terms
removing features
regularization techniques
using a non-linear model
3. Decision Tree

Now, this is one of my favorite algorithms. It is a type of supervised learning algorithm that is mostly used for classification problems. Surprisingly, it works for both categorical and continuous dependent variables. In this algorithm, we split the population into two or more homogeneous sets. This is done based on the most significant attributes/ independent variables to make as distinct groups as possible.

In the image above, you can see that the population is classified into four different groups based on multiple attributes to identify ‘if they will play or not.

4. Naive Bayes

This is a classification technique based on Bayes’ theorem with an assumption of independence between predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier would consider all of these properties to independently contribute to the probability that this fruit is an apple.

Naive Bayesian model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

Bayes theorem provides a way of calculating posterior probability P(c|x) from P(c), P(x), and P(x|c). Look at the equation below:

Here,

P(c|x) is the posterior probability.
P(c) is the prior probability of class.
P(x|c) is the likelihood.
P(x) is the prior probability of the predictor.
Example:

Let’s understand it using an example. Below I have a training data set of weather and corresponding target variable ‘Play’. Now, we need to classify whether players will play or not based on weather conditions. Let’s follow the below steps to perform it.

1: Convert the data set to the frequency table

2: Create a Likelihood table by finding the probabilities like Overcast probability = 0.29 and probability of playing is 0.64

3: Now, use the Naive Bayesian equation to calculate the posterior probability for each class. The class with the highest posterior probability is the outcome of the prediction.

Problem:

Players will pay if the weather is sunny, is this statement is correct?

We can solve it using above discussed method, so P(Yes | Sunny) = P( Sunny | Yes) * P(Yes) / P (Sunny)

Here we have P (Sunny |Yes) = 3/9 = 0.33, P(Sunny) = 5/14 = 0.36, P( Yes)= 9/14 = 0.64

Now, P (Yes | Sunny) = 0.33 * 0.64 / 0.36 = 0.60, which has higher probability.

Naive Bayes uses a similar method to predict the probability of different classes based on various attributes.

[code]
# import libraries
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

dataset = datasets.load_iris()

#create model
model = GaussianNB()
model.fit(dataset.data, dataset.target)

#make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
[/code]
5. kNN (k- Nearest Neighbors)

kNN used for both classification and regression problems. We use kNN in classification problems in the industry. K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases by a majority vote of its k neighbors

These distance functions can be Euclidean, Manhattan, Minkowski, and Hamming distance. At times, choosing K turns out to be a challenge while performing kNN modeling.

We can make kNN to our real life. If you want to learn about a person, of whom you have no information, you might like to find out about his close friends and the circles he moves in and gain access to his/her information!

[code]

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Loading data
irisData = load_iris()

# Create feature and target arrays
X = irisData.data
y = irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size = 0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print(knn.predict(X_test))

[/code]
Things to consider before selecting kNN:
KNN is computationally expensive
Normalize Variables, else higher range variables can bias it
Works on pre-processing stage more before going for kNN like an outlier, noise removal

This brings me to the end of this blog. Stay tuned for more content on Machine Learning and Data Science!

Got a question for us? Please mention them in the comments section and we will get back to you.
