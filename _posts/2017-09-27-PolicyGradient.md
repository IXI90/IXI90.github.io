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

In general, we will try to keep the notation of the "Introduction to Reinforcement Learning" blog post series. But, let us recall our RL framework real quick and change a few small things. As before we want to represent our environment as a Markov Decision Process (MDP) $< \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$, which comprises: a state space $\mathcal{S}$, an action space $\mathcal{A}$, a stationary transition distribution function $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \[ 0,1 \ ]$, a reward function $\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ and a discount factor $\gamma \in \[ 0,1 \]$.      

