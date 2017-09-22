---
layout: post
comments: true
title:  "Policy Gradient Methods"
excerpt: "In this post the key concepts behind Policy Gradient Methods will be discussed. Sample topics are the REINFORCE algorithm and the Policy Gradient Theorem. In the end, the learned algorithms will be used to solve the MountainCar environment of the OpenAI Gym."
date:   2017-09-27
mathjax: true
---

## Value-based vs. Policy-based Methods

Most of the famous success stories of Reinforcement Learning (RL) were in the area of value-based methods, e.g. Deep Q-Networks (Q-algorithm). As we have seen before, those methods primarily try to estimate the optimal value or optimal action-value function. An optimal policy is then derived from the value function (e.g. by greedy action). In the context of policy-based methods one follows a different strategy. Here, the policy is parametrised and thus, approximated directly without estimating the related value functions first. In other words, instead of learning an approximation of the underlying value function, policy-based methods attempt to search the space of possible policies directly. Why is this a good idea?

Actually there are a lot of different reasons why this strategy might be beneficial: First of all, in certain environments the optimal policy might just be easier to approximate than the action-value function. However, this is really difficult to evaluate beforehand. A second reason is the possibility to represent stochastic policies. Most implementations of value-based methods lead to rather deterministic policies and thus ignore a lot of other possible good candidates. For example, in rock-paper-scissors with an "intelligent" opponent the Q-algorithm would not yield a promising strategy - since every deterministic policy will be beaten by the opponent after a few episodes. 
Furthermore, a direct parametrisation of the policy has the nice effect that small changes in the parameters of the function lead to small changes in the policy. This was not at all true for value-based algorithms, due to the nature of greedy-policies. 

One additional remark: Above we presented value-based and policy-based methods as two separated opposing sides. However, those methods are not entirely disjoint. There are methods, called actor-critic models, which utilize value-based and policy-based ideas at the same time. Those actor-critic methods parametrise the policy directly and evaluate the quality of the policy with value-based methods.

## Our Situation

In general, we will try to keep the notation and setting of the "Introduction to Reinforcement Learning" blog post series. But, let us recall our RL framework real quick and change a few small things. As before we want to represent our environment as a Markov Decision Process (MDP) $< \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$, which comprises: a state space $\mathcal{S}$, an action space $\mathcal{A}$, a stationary transition distribution function $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \[ 0,1 \]$, a reward function $\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ and a discount factor $\gamma \in \[ 0,1\]$. Instead of $\mathcal{P} (s,a,s')$ we will often write $\mathcal{P}(s'\mid s,a)$ to express the probability of ending up in state $s'$ after taking action $a$ in state $s$. In this MDP the agent sequentially chooses actions over a sequence of discrete time steps. 

As mentioned above, we want to facilitate stochastic policies this time, thus a policy is now defined as a function $\pi: \mathcal{S} \rightarrow Prob(\mathcal{A})$, where $Prob(\mathcal{A})$ is the set of all probability measures on $\mathcal{A}$. We write $\pi(a \mid s)$ (instead of $\pi(s)(a)$) to express the probability of taking action $a$ in state $s$ under the policy $\pi$. 
Let us assume that we are in an episodic setting with a fixed endpoint at time step $T$, i.e. every possible trajectory is of the form $<s_0,a_0,s_1,a_1,...,s_T>$ with $s_i \in \mathcal{S}, a_i \in \mathcal{A}$. However, this does not mean that the agent can not reach an absorbing state before time step $T$. The start state $s_0$ of an episode is usually determined by an initial state distribution $\mathcal{P}_0$, i.e. for a discrete state space $\mathcal{P}_0 (s)$ gives us the probability that an episode starts in state $s$. To simplify our notation, we will often just assume that there exists one designated start state $s_0 \in \mathcal{S}$, thus $\mathcal{P}_0 (s_0)=1$. Now, the transition function $\mathcal{P}$ together with a policy $\pi$ defines a probability measure $\mathcal{P} _{\pi}$ on the set $\mathbb{T}:= \mathcal{S} \times \mathcal{A} \times ... \times \mathcal{S}$ of all possible trajectories (set of trajectories is equipped with the Borel $\sigma$-algebra induced by the product topology). This probability measure is defined by $\mathcal{P} _{\pi} (<s_0,a_0,s_1,...,s_T>)= \mathcal{P} _0 (s_0) \prod _{t=0}^{T-1} \pi(a _t \mid s _t) \mathcal{P}(s _{t+1} \mid a_t,s_t)$ for an arbitrary trajectory $<s_0,a_0,s_1,...,s_T> \in \mathbb{T}$. With respect to this measure we can now again define the state-value function 

$$ V _{\pi} (s) := \mathbb{E} _{\pi} [ \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t} \mid s _0 = s ] = \int _{\mathbb{T}_s} \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t}(\tau)  \mathcal{P} _{\pi} (\tau) d \tau  $$, 

where $R_t: \mathbb{T} \rightarrow \mathbb{R}$ with $R_t (<s_0,a_0,s_1,...,s_T>) := \mathcal{R} (s_t,a_t)$ and $\mathbb{T}_s:= \langle <s_0,a_0,s_1,a_1,...,s_T> \in \mathbb{T} \mid s_0 = s \rangle$. This value function gives us the expected return of an episode that starts in state $s$ and where the agent then follows policy $\pi$. From now on always assume that we have one designated start state $s_0$ ($\mathcal{P}_0 (s_0) = 1$), i.e. we are primarily interested in $V _{\pi} (s_0)$.

## Policy Gradient Methods

In this class of algorithms the policy gets parametrised directly, thus we have a collection of parameters $\theta \in \mathbb{R}^n$ and those parameters define a policy $\pi _{\theta}$. For example, the parameters $\theta$ could be the weights of an artificial neural network and the values of the output units could represent the policy $\pi _{\theta}$. In all those algorithms we will make sure that $\pi _{\theta}$ is differentiable with respect to $\theta$. 
Our goal is to improve the policy $\pi _{\theta}$ by changing the underlying parameters $\theta$. But how do we evaluate whether $\pi _{\theta}$ is a good or bad policy?
      
For this we want to use the concept of the state-value function! We define $J(\theta) := V_{\pi_{\theta}} (s_0)$ as our objective function, which we then try to maximize. In other words, one tries to find suitable parameters $\theta$ that maximize the function $J$. As seen in the past blog posts, an optimal policy would yield a suprememum of $J$. It should be noted that this start-value in an episodic environment is only one way to define an objective function. In the literature you will find different strategies, e.g. average value and average reward per time-step. However, the resulting algorithms are often very similar.
 
 Like so many times before in machine learning, we would like to use gradient ascent to maximize the objective, i.e. we want to change $\theta$ in the direction of $\nabla _{\theta} J(\theta)$. At first glance this seems problematic, because $J$ contains an integral with respect to the probability measure $\mathcal{P} _{\pi _{\theta}}$, which in turn is influenced by the transition dynamics of the MDP and those dynamics are usually unknown.
 
 Let us try to calculate the gradient anyway...
 
 $$ \nabla _{\theta} J(\theta) = \nabla _{\theta} V _{\pi _{\theta}} (s_0) = \nabla _{\theta} \int _{\mathbb{T}} \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t}(\tau)  \mathcal{P} _{\pi _{\theta}} (\tau) d \tau \\ =  \int _{\mathbb{T}} \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t}(\tau)   \nabla _{\theta} \mathcal{P} _{\pi _{\theta}} (\tau) d \tau = \int _{\mathbb{T}} \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t}(\tau) \mathcal{P} _{\pi _{\theta}} (\tau)  \nabla _{\theta} log( \mathcal{P} _{\pi _{\theta}} (\tau)) d \tau $$
 
(I'm aware of the fact that we are not addressing some mathematical issues - like measurability and rules for interchanging integral and derivative. The goal of this post is rather to give a quick and "dirty" introduction to this topic.)

The usage of the derivative of the logarithm function in this context (in the last step above) is often called the "log-likelihood trick" in the literature. Now, consider for $\tau = <s_0,a_0,s_1,...,s_T>$ that 

$$ \nabla _{\theta} log( \mathcal{P} _{\pi _{\theta}} (\tau)) = \nabla _{\theta} log( \prod _{t=0}^{T-1} \pi _{\theta} (a _t \mid s _t) \mathcal{P}(s _{t+1} \mid a_t,s_t)) \\ = \nabla _{\theta} ( \sum _{t=0}^{T-1} log(\pi _{\theta} (a _t \mid s _t)) + \sum _{t=0}^{T-1} log(\mathcal{P}(s _{t+1} \mid a_t,s_t))) = \sum _{t=0}^{T-1}  \nabla _{\theta} log(\pi _{\theta} (a _t \mid s _t)) $$

Here, we simply used the facts that the logarithm transforms products into sums and that the transition dynamics of the MDP do not depend on our parameters $\theta$. Let us now simply insert this result into the equation from above...

$$ \int _{\mathbb{T}} \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t}(\tau) \mathcal{P} _{\pi _{\theta}} (\tau)  \nabla _{\theta} log( \mathcal{P} _{\pi _{\theta}} (\tau)) d \tau = \int _{\mathbb{T}} (\sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t}(\tau))  (\sum _{t'=0}^{T-1}  \nabla _{\theta} log(\pi _{\theta} (a _t' \mid s _t'))) \mathcal{P} _{\pi _{\theta}} (\tau) d \tau \\ = \mathbb{E} _{\pi _{\theta}} [ \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t} \sum _{t'=0}^{T-1}  \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'}) ) ] $$

A parameter update in the direction of $ \mathbb{E} _{\pi _{\theta}} [ \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t} \sum _{t'=0}^{T-1}  \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'}) ]$ is often called (episodic) REINFORCE algorithm. The nice thing about this expression of the gradient $\nabla _{\theta} J(\theta)$ is that we were able to sample this expected value in a very intuitive way (Monte Carlo estimate). Assume we experienced trajectories $<s_0,a_0^i,r_0^i,s_1^i,a_1^i,r_1^i,...,s_T^i>$ for $i = 1,...,M$ by applying policy $\pi _{\theta}$. As a consequence,

$$ \nabla _{\theta} J(\theta) \approx 1/M \sum _{i=1}^{M} \sum _{t=0}^{T-1} \gamma ^t r _{t}^i \sum _{t'=0}^{T-1}  \nabla _{\theta} log(\pi _{\theta} (a _{t'}^i \mid s _{t'}^i) )$$

This is quite cool, but we want to go even a little bit further and try to construct gradient estimators that tend to have a lower variance than the one above. This time one uses $\nabla _{\theta} J(\theta) = \nabla _{\theta} \mathbb{E} _{\pi _{\theta}} [ \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t}] =  \sum _{t=0}^{T-1} \gamma ^t \nabla _{\theta} \mathbb{E} _{\pi _{\theta}} [ \mathcal{R} _{t}] $. Additionally, let us define the set of all possible trajectories until time step $t$, denoted by $\mathbb{T}_t := \langle <s_0,a_0,...,s_t,a_t,s _{t+1}> \mid s _i \in \mathcal{S}, a _i \in \mathcal{A} \rangle$ and probability measure $\mathcal{P} _{\pi _{\theta}}^t (<s_0,a_0,s_1,...,s _{t+1}>)= \prod _{t=0}^{t} \pi _{\theta} (a _t \mid s _t) \mathcal{P}(s _{t+1} \mid a_t,s_t)$. This implies:

$$\nabla _{\theta} \mathbb{E} _{\pi _{\theta}} [ \mathcal{R} _{t}] =  \nabla _{\theta} \int _{\mathbb{T}} \mathcal{R} _{t}(\tau)  \mathcal{P} _{\pi _{\theta}} (\tau) d \tau \\ =  \nabla _{\theta} \int _{\mathbb{T}_t} \mathcal{R} _{t}(\tau_t)  \mathcal{P} _{\pi _{\theta}}^t (\tau) d \tau_t = ... = \int _{\mathbb{T}_t} \mathcal{R} _{t}(\tau_t)  (\sum _{t'=0}^{t}  \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'}))) \mathcal{P} _{\pi _{\theta}}^t (\tau_t) d \tau_t \\ =  \int _{\mathbb{T}} \mathcal{R} _{t}(\tau)  (\sum _{t'=0}^{t}  \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'}))) \mathcal{P} _{\pi _{\theta}} (\tau) d \tau = \mathbb{E} _{\pi _{\theta}} [ \mathcal{R} _{t}  (\sum _{t'=0}^{t}  \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'}))) ]  $$  

Now, we insert this into the gradient formula:

$$ \nabla _{\theta} J(\theta) = \sum _{t=0}^{T-1} \gamma ^t  \mathbb{E} _{\pi _{\theta}} [ \mathcal{R} _{t}  \sum _{t'=0}^{t}  \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'})) ] \\ = \mathbb{E} _{\pi _{\theta}} [ \sum _{t=0}^{T-1} \gamma^t \mathcal{R}_t \sum _{t'=0}^{t}  \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'}))] = \mathbb{E} _{\pi _{\theta}} [ \sum _{t'=0}^{T-1} \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'})) \sum _{t=t'}^{T-1} \gamma^{t} \mathcal{R}_t ] \\ = \mathbb{E} _{\pi _{\theta}} [ \sum _{t'=0}^{T-1} \gamma^{t'} \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'})) \sum _{t=t'}^{T-1} \gamma^{t-t'} \mathcal{R}_t ] $$

This expression of the policy gradient is often the main statement of the "Policy Gradient Theorem". Again we can use this to develop a nice and simple Monte Carlo method to approximate the gradient of the objective function. As before, assume we experienced trajectories $<s_0,a_0^i,r_0^i,s_1^i,a_1^i,r_1^i,...,s_T^i>$ for $i = 1,...,M$ by applying policy $\pi _{\theta}$. Then one can approximate the gradient by...    

$$ \nabla _{\theta} J(\theta) \approx 1/M \sum _{i=1}^{M} \sum _{t'=0}^{T-1} \gamma^{t'} \nabla _{\theta} log(\pi _{\theta} (a _{t'}^i \mid s _{t'}^i)) \sum _{t=t'}^{T-1} \gamma^{t-t'} r_t^i $$

Please, note that $\sum _{t=t'}^{T-1} \gamma^{t-t'} r_t^i$ is basically an estimate of the action-value function of $\pi _{\theta}$ at time step $t$. At this point we could go into the direction of actor-critic methods by using an estimate of this action-value function, instead of sampling whole trajectories. But let us postpone this, and instead put the just derived algorithm into pseudocode:

<img src="https://raw.githubusercontent.com/IXI90/IXI90.github.io/master/REINFORCE.jpg" width="600" height="180" />

*Remark*: If one is still worried about the variance of our gradient estimate, an additional "baseline" should be considered. The idea behind this concept is the following: Let's say we have a random variable $X$ with a rather high variance $Var(X)$, i.e. in our case the gradient of the objective function $J$. Then it would be beneficial to find a random variable $B$  with the properties...
 1. $E[X-B]=E[B]$ 
 2. $Var(B)$ small 
 3. $Cov(X,B)$ high. 
 
 This implies that $Var(X-B)= Var(X)+Var(B)-2Cov(X,B) < Var(X)$ and hence, one can simply try to estimate $X-B$ instead of $X$ directly. Due to condition (1) this random variable does not change the desired expected value, but has a reduced variance. Now, back to our situation. With a few rather easy steps (comparable to those above), one can show that for any function $b: \mathcal{S} \rightarrow \mathbb{R}$ one has $\mathbb{E}_{\pi _{\theta}} [\nabla _{\theta} log(\pi _{\theta} (a_t \mid s_t)) b(s_t)] = 0$ and thus, condition (1) is always fulfilled:
 
 $$ \mathbb{E} _{\pi _{\theta}} [ \sum _{t'=0}^{T-1} \gamma^{t'} \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'})) \sum _{t=t'}^{T-1} \gamma^{t-t'} \mathcal{R}_t ] = \mathbb{E} _{\pi _{\theta}} [ \sum _{t'=0}^{T-1} \gamma^{t'} \nabla _{\theta} log(\pi _{\theta} (a _{t'} \mid s _{t'})) ( \sum _{t=t'}^{T-1} \gamma^{t-t'} \mathcal{R}_t - b(s_t) )]$$ 
 
The crucial point here is to use the fact that the baseline function $b$ does not depend on the action space $\mathcal{A}$. In practice, the baseline function $b$ is often equal to the state-value function, i.e. the expected return following policy $\pi _{\theta}$ from the given state until terminal time step $T$. 

## Example

One very famous testing domain in RL is the Mountain Car problem. Here, the agent controls a small, under-powered car in a 2D-world. At the beginning of an episode the car is placed in a valley (designated start state $s_0$) and the goal is to get on top of the rightmost hill. Due to the weak engine of the car, the agent can not directly accelerate up the steep slope of the right hill. Instead it has to first drive up the opposite hill and then use the additional power of the downswing to get to the target location.

 <img src="https://raw.githubusercontent.com/IXI90/IXI90.github.io/master/MountainCar.png" width="300" height="170" />

The aim is now to apply the algorithm described above with pseudocode to this problem. Every state of this (OpenAI Gym) environment is represented by two scalar values, namely position $P \in \[-1.2,0.6 \] and velocity $V \in \[ -0.07,0,07 \]$, i.e. we have a continuous state space. At every time step the agent can choose between three actions - push left, no push and push right.
Besides, every episode 