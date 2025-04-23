# Reinforcement Learning of Longitudinal Searching

This repository provides an example of applying Reinforment Learning(RL) method to longitudinal planning for autonomous vehicles.

## RL Problem Construction

### Environment

For simplicity, longitudinal planning focuses on filling kinetics on a given path in the planning horizon. We could use the inputs and constraints of the traditional longitidinal planning problem to define the environment for RL. 

1. ST polygons of the whole horizon;
2. Speed limits of the entire path;
3. Kinetics of a single step.

### Action Space

In this repository, we select continuous **jerk**(the derivative of acceleration) as the only action.

### Observation Space

In a single planning frame, we could encode what being observed as the input features. As discussed in the RL Environment, we could encode information by the following way.

1. ST polygon: encoded as images
2. Speed limits and Kinetics: encoded as arrays

### Rewards

We use the negtive costs as rewards.

1. Station Cost
    
    - The S-T curve should **NOT** overlap with hard ST polygons, otherwise a large penalty will be given. 
    - If the agent cross a hard ST polygon at step $t$,  extra penalty will be applied in the rest $T - t$ steps.

2.  Speed Cost

    - The speed at any step should **NOT** be negative.
    - The agent should **NOT** overspeed. Also, extra penalty will be punished for the rest steps once the agent overspeeds at the first step.
    - The agent should keep at the cruise speed as close as possible.

3. Smooth Cost

    - Punish large absolute accelerations or those out of range scope.
    - Punish large jerks
    - Punish the derivatives of jerk

4. Interative Balance

    To make the consistence of gradient descending, weights of costs should be dynamically adjusted according to environments. For example,

    1. Cost of agent deviating the cruise speed should decay when it interacts with ST polygons;
    2. Cost of smoothness shoule decrease when the agent crosses a ST polygon.
   
   Additionally, the shape of S-T, V-T, A-T, J-T curves could also be punished at the ternimate state, such as

      1. the slope of s-t curve where it cross an ST polygon;
      2. oscillation of j-t curve (may be mitigated by punish d_j).

### Agent

The RL problem is trained by Soft Actor-Critis(SAC).