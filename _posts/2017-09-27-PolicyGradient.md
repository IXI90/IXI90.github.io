---
layout: post
comments: true
title:  "Policy Gradient Methods"
excerpt: "In this post the key concepts behind Policy Gradient Methods will be discussed. Sample topics are the REINFORCE algorithm and the Policy Gradient Theorem. In the end, the learned algorithms will be used to solve the MountainCar environment of the OpenAI Gym."
date:   2017-09-27
mathjax: true
---

## Policy Gradient Methods

Most of the famous success stories of Reinforcement Learning (RL) were in the area of value-based methods, e.g. Deep Q-Networks. As we have seen before those methods primarily try to estimate the optimal value or action-value function. A optimal policy is then derived from the value function (greedy action). 
