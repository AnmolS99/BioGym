# BioGym 🐾

BioGym is a spatio-temporal wildlife management RL environment created on the [Gymnasium](https://gymnasium.farama.org/) library (formerly OpenAI Gym).

It was created as part of my Master's thesis (TDT4900), which I wrote spring 2023 at NTNU under the superivision of Keith L. Downing. My thesis is available [here](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3097883).

I presented my thesis at the NorwAI Innovate Conference 2023, and also had a poster, which you can see [here](https://www.ntnu.edu/documents/1294735055/1345620514/03_Anmol.pdf/fd93b5cf-4c83-5fa4-66c6-b545ec999b05?t=1706521849112).

## Master's thesis 🎓

My thesis focused on *Deep Reinforcement Learning for Spatio-Temporal Wildlife Management*, and my research goal was to:

*Explore the use of different deep reinforcement learning (DRL) algorithms on the task of spatio-temporal wildlife management, with the aim of maintaining a diverse and stable ecosystem.*

To do this, I created a spatio-temporal wildlife management simulation, based on a tri-trophic predator-prey model. This was then wrapped as a Gymnasium environment, so that it followed the standard API of the library. This made it possible to "plug-and-play" with RL algorithms from [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/). It also makes it easier for those interested to work with this RL environment.

I trained and tested three different RL algorithms on the RL environment (DQN, A2C and PPO), which gives a higher reward for more biodiversity. Different metrics were used to measure biodiversity. The RL agent had the possibility to remove or add population of one of the three species at each timestep.

## Overview

Below is an overview of the system designed to investigate the applicability of DRL on a spatio-temporal wildlife management simulation:

<img alt="Wildlife Environment Overview" src="https://github.com/AnmolS99/BioGym/assets/6755688/c09b5a9d-a3b9-40df-8eba-bbbd0c34d916" width="800">

## Results

While all RL algorithms were able to improve with training, PPO standed out with stable performance and steady increase in performance. Below is one of the results gathered when training the algorithms on BioGym:

<img alt="DRL algorithm preformance on a 2x2 grid with 10x action multiplier - Equal training time" src="https://github.com/AnmolS99/BioGym/assets/6755688/b044c80a-a2de-420b-8caa-cd26411ae6ab" width="600">

Interestingly, the RL algorithms employed quite different policies. Some would always remove population of one species, while others both removed and added populations. In my thesis I hypothesize how the policy each algorithm learns are connected to the algorithm's design.

For details on research background and goal, related work, research process, and results - please take a look at my [thesis](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3097883).
