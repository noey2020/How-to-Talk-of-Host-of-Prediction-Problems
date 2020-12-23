# How-to-Talk-of-Host-of-Prediction-Problems

December 22, 2020

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

- Our very first concrete machine learning problem
was classifying handwritten digits.
This is an example of a prediction problem.
What I want to do today is to take a step back
and to talk about prediction problems
in a slightly more general and abstract way.
So, we'll begin with a few remarks
about the difference between machine learning
and algorithms,
I'll then formalize the notion
of a prediction problem
and talk about a way in which we can organize this space,
and finally, I'll end with a roadmap
of the rest of this class.
Okay, machine learning versus algorithms.
So, in today's computer-driven world,
two core technologies are machine learning
and algorithms.
What is the similarity or difference between these?
Well, interestingly, it turns out
that both fields have a common central goal
and that is to develop procedures,
little pieces of code,
that exhibit the desired input/output functionality.
In algorithms, for example,
people have spent decades looking for good ways
to find shortest paths in graphs.
So, the input is a graph along with two nodes in it,
and the desired output is the shortest path
between those two nodes.
An algorithm is a precise set of steps
that leads from the input to the desired output.
In machine learning,
we also want to come up with procedures
that go from an input to a desired output,
but the sort of problems that we're dealing
with are of a different nature
so that it is very hard to list a precise set of steps
that will get us to that output.
Okay, so for example,
let's say the input is the picture of an animal
and the desired output is the name of the animal.
What are some precise steps that will do this for us?
It's hard to imagine,
in fact, the problem isn't even precisely defined
and so, rather than try and come up
with the algorithm ourselves,
what we do is to collect a whole bunch
of XY examples of input/output pairs
and then we ask the machine
to figure out a mapping on its own.
So, this is a prediction problem.
There is an input space, we'll always call it X,
for example, the space of all images of animals,
and then there's an output space,
for example, names of animals.
We have in mind some
desired mapping from inputs to outputs,
but it's not something that's possible
for us to necessarily specify in a precise way
and so we collect a training set of XY examples.
The learning machine takes this training set
and uses it to pick a mapping from X to Y.
A function that takes the image of an animal
and returns the name of the animal,
or it returns the name that it believes to be correct.
Typically, the way a learning algorithm works is
by looking for a function, a mapping,
that does well on the training set.
Now, there are many, many different types
of prediction problems out there
and one way in which it's common
to categorize them is according to the type
of output space,
and there are three cases that we typically distinguish.
When the output space is discrete,
when it's continuous,
and when it consists of probability values.
It turns out that these three cases require
somewhat different methods.
So, let's look at the first case.
So, this is when the outputs are discrete
and this is a case that we call classification.
The simplest setting is binary classification
where there are just two possible outputs,
good or bad, plus or minus, yes or no.
In spam detection, for example,
the inputs X are email messages
and the desired output is just spam or not spam.
There are just two choices, it's binary.
Very often, in classification problems,
there are more than two possible outputs.
So, for instance, maybe the input is a news article
and the desired output is the subject matter
of the article,
politics, business, technology,
sports, entertainment, et cetera.
So, that's called multiclass classification
and there are also cases
in which the desired output is discrete
but it's something more complex
that has some combinatorial structure to it.
In parsing, for example, the input is a sentence,
like John hit the ball,
and the desired output is the parse tree
for that sentence,
so this entire object over here.
So, the output space is still finite,
it's still discrete but it's more complex
and it has a certain kind of structure to it.
What we'll be doing is devoting a large amount
of attention to binary classification
because this is a nice and simple setting
in which to study prediction problems.
It'll then turn out that the methods we develop
can generalize quite easily
to the other two settings as well,
to multiclass and structured outputs.
Okay, so we've been talking about
categorizing prediction problems
by the type of output space,
so when the outputs are discrete,
we call it a classification problem.
When the outputs are continuous,
then we have a regression problem.
Let's look at a couple of examples.
So, the first example here is,
suppose we wanna predict the pollution level tomorrow
and we're interested in this
because, for instance, it will help us decide
whether we are gonna let the kids go out tomorrow
or whether we should keep them indoors.
So, how does one measure pollution level?
Well, one common way of doing it is
by something called the air quality index,
so this is a number, you know.
A positive number that's less than a hundred
means the air quality is not too bad.
If it's more than a hundred,
that means that it's not good,
and if it's more than 200,
it means that the air is absolutely dangerous.
So, the output space now consists
of positive numbers, it's a continuous space.
Even if we only predict integer values,
we still think of it as a continuous space
because, for example, a prediction of a hundred
is very close to a prediction of 101
or a prediction of 102,
and is very far from a prediction of 200.
In other words, the Ys lie on a scale.
Let's look at another example.
Insurance company calculations.
So, one of the things that an insurance company
is interested in when you apply for a policy is
how much longer do they expect you to live,
and this determines, for example,
how much they would charge you.
So here, the number we wanna predict, the Y,
is the age at which you're going to die.
Let's say it's something in the range zero to 120
and again, this is something that we could always round
to the nearest integer,
but we still think of it as a continuum
because these numbers lie along a scale.
Now, one interesting question is
what there is to sort of think about is,
what are suitable predictor variables
for these kinds of regression problems?
Okay, so for example, life expectancy.
What are the sort of variables
or what are the sort of pieces of information
that might be helpful in determining this?
Well, this is actually a very well-studied area
and the sort of information
that people use is your age,
your gender, women tend to live longer,
whether you smoke or not,
do you have high blood pressure
and are you taking medicine for it,
do you have high cholesterol
or are you taking medicine for it,
do you smoke,
and a few other such things.
Okay.
So, the final kind of output space we'll look at is
when the Ys represent probabilities.
So here, the output space
is literally the range zero to one.
Now, this seems a little bit like regression
because it is a continuous range,
but it turns out that this particular case
does require specialized methods
and so we tend to treat it separately.
So, an example here is credit card fraud detection,
so X, the input, consists of the details
of a credit card transaction.
What is the amount of the purchase?
What is being bought?
What is the name of the merchant?
What is the zip code?
And so on.
And Y here, the output we wish to predict,
is what is the probability
that this transaction is fraudulent?
So, this is a probability estimation problem.
One interesting question over here is,
why not just think of this as binary classification?
Why are we trying to predict the probability?
Why not just say there are two possible labels,
fraudulent or legitimate,
and that's what we wanna predict?
So, that's a good question.
It turns out that the reason we wanna use probabilities is
because the results of this prediction are gonna be used
as part of a larger decision-making framework
and it's just one of the pieces of information
that go into the decision.
So, for instance,
if a transaction has got a high probability
of being fraudulent,
then of course the transaction should be denied,
but if the transaction has a low probability
of being fraudulent,
should it be denied or not?
If it has just a small but significant probability
of being fraudulent,
should we accept it or deny it?
Well, it might depend on other factors
like the amount of the transaction.
If it's just a small transaction,
then maybe it's okay to take the risk.
If it's a large transaction,
then it should probably be denied.
So, there are many factors that go into the decision
and in order to assess the risks correctly,
it's very useful to have not just a binary prediction,
fraudulent or legitimate,
but an actual probability value.
Okay, so we've been talking a lot about prediction problems
and indeed, this is the bread and butter
of machine learning,
and we'll be spending a lot of time on this in the course.
Is there anything other than prediction problems?
Well, very often
we see the dataset
and we want to understand it better,
and don't necessarily have a specific
prediction task in mind.
We just want to know,
is there interesting structure in the data?
Are there clusters in it, for example?
So, the sort of topics that come under this,
which we can call representation learning,
are things like clustering, projection,
dictionary learning, and so on.
And we'll spend a little bit of time on this
and we'll end with deep learning,
which is, in a sense, a way of combining
both representation learning
and prediction problems.
Okay, well, that is a little bit
of an overview of the course.
Hopefully it's clear what lies ahead
and next item we'll begin
with the systematic treatment of classification.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-of-Useful-Distance-Functions

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
