# Wellcome to DeepRL repository!
## What is this project supposed to do? What problem does it solve?
	It provides code for creating *agent* entities that are capable of \
interacting with an *environment* (provided by user) in a manner that \
humans would call *reasonable* behaviour. But what does it actually mean?
Well, that means that our *environment* (bunch of states and transition \
rules) has to be
1) *observed* (fully or partially) at any given moment of time,
2) *acted on* (in a finite or infinite number of ways) at any given moment \
of time.
That also means that we are generating a set of rules which ouput \
*actions* when provided with *observations*.
	Now wait a minute, if we are to have a computer program that interacts \
with environment (instead of us doing it manually) we want those actions to be \
meaningful. We don't want them to be stupid actions, don't we? Anyone can \
output stupid actions. And how would we ensure that those actions are \
meaningful? Well, for that reason we better have our environment to carry a
3) fitness function, usually implemented via *reward*.
It provides user with some feedback. Big rewards mean we are doing okay, \
small rewards mean we have to try harder (or rather our agent has to try \
harder).
	Good thing mathematicians came up with algorithms that solve exactly this \
problem. *Reinforcement learning*: *agent* interacts with an *environment* \
and maximizes expected *reward* over time.
Back to code.
### More specifically,
we want to have mechanisms for **training** agents, **evaluating** agent's \
performance and **saving** agent's behavior for further use. This is what \
this code helps with. It is also designed to allow carrying out \
experiments: comparing different agents in various ways.
	If you want to use this code in your project, checking out \
(examples)[examples/] is a good start. DeepRL doesn't have any \
documentation (and won't have any time soon).
	Better install these python packages or it won't work:
- pytorch
- numpy
- pandas
- matplotlib

### Good luck and have fun!
