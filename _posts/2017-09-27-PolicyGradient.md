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

One additional remark: Above we presented value-based and policy-based methods as two separated opposing sides. However, those methods are not entirely disjoint. There are methods, called actor-critic models, which utilize value-based and policy-based ideas at the same time. Those actor-critic methods parametrise the policy directly and evaluate the quality of the policy with value-based methods. An example of this class of algorithms will be presented later on.

## Our Situation

In general, we will try to keep the notation and setting of the "Introduction to Reinforcement Learning" blog post series. But, let us recall our RL framework real quick and change a few small things. As before we want to represent our environment as a Markov Decision Process (MDP) $< \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$, which comprises: a state space $\mathcal{S}$, an action space $\mathcal{A}$, a stationary transition distribution function $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \[ 0,1 \]$, a reward function $\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ and a discount factor $\gamma \in \[ 0,1\]$. Instead of $\mathcal{P} (s,a,s')$ we will often write $\mathcal{P}(s'\mid s,a)$ to express the probability of ending up in state $s'$ after taking action $a$ in state $s$. In this MDP the agent sequentially chooses actions over a sequence of discrete time steps. 

As mentioned above, we want to facilitate stochastic policies this time, thus a policy is now defined as a function $\pi: \mathcal{S} \rightarrow Prob(\mathcal{A})$, where $Prob(\mathcal{A})$ is the set of all probability measures on $\mathcal{A}$. We write $\pi(a \mid s)$ (instead of $\pi(s)(a)$) to express the probability of taking action $a$ in state $s$ under the policy $\pi$. 
Let us assume that we are in an episodic setting with a fixed endpoint at time step $T$, i.e. every possible trajectory is of the form $<s_0,a_0,s_1,a_1,...,s_T>$ with $s_i \in \mathcal{S}, a_i \in \mathcal{A}$. However, this does not mean that the agent can not reach an absorbing state before time step $T$. The start state $s_0$ of an episode is usually determined by an initial state distribution $\mathcal{P}_0$, i.e. for a discrete state space $\mathcal{P}_0 (s)$ gives us the probability that an episode starts in state $s$. To simplify our notation, we will often just assume that there exists one designated start state $s_0 \in \mathcal{S}$, thus $\mathcal{P}_0 (s_0)=1$. Now, the transition function $\mathcal{P}$ together with a policy $\pi$ defines a probability measure $\mathcal{P} _{\pi}$ on the set $\mathbb{T}:= \mathcal{S} \times \mathcal{A} \times ... \times \mathcal{S}$ of all possible trajectories (set of trajectories is equipped with the Borel $\sigma$-algebra induced by the product topology). This probability measure is defined by $\mathcal{P} _{\pi} (<s_0,a_0,s_1,...,s_T>)= \mathcal{P} _0 (s_0) \prod _{t=0}^{T-1} \pi(a _t \mid s _t) \mathcal{P}(s _{t+1} \mid a_t,s_t)$ for an arbitrary trajectory $<s_0,a_0,s_1,...,s_T> \in \mathbb{T}$. With respect to this measure we can now again define the state-value function 

$$V _{\pi} (s) := \mathbb{E} \[ \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t} \mid s_0 = s\] = \int _\mathbb{T}_s \sum _{t=0}^{T-1} \gamma ^t \mathcal{R} _{t}(\tau)  \mathcal{P} _{\pi} (\tau) d \tau  $$, 

where $R_t: \mathbb{T} \rightarrow \mathbb{R}$ with $R_t (<s_0,a_0,s_1,...,s_T>) := \mathcal{R} (s_t,a_t)$ and $\mathbb{T}_s:= \{<s_0,a_0,s_1,a_1,...,s_T> \in \mathbb{T} \mid s_0 = s \}$. This value function gives us th        
      

