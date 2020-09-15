# 6. Knapsack and Basics of Optimization

Data Science = Data Engineering + Data Analytics + <u>Decision Making</u>

- Data-driven Optimization(Prediction-driven Optimization)
  - Decision based on prediction, not on past statistical pattern

## Real-World Example(Market Share Prediction)

- Airline A가 4가지 노선을 운영하는데, <u>flight frequency를 adjust</u>하여 <u>한정된 budget</u>으로 <u>최대 profit</u>을 낼 것!
- 성공적인 결과 도출을 위한 2가지 요소
  - **Precise Forecasting**(3월에 LA to  NY로 700회의 flight를 배정 $\to$ \$10M profit)
    - Pre-trained model
  - **Optimal Allocation**(적절한 frequency를 배정)
  - Optimization Module에서 도출한 optimized frequency가 다시 Forecasting ML Model에 들어가면서 결과적으로 maximize된 budget을 구할 수 있음
  - "Train the ML model" $\to$ "Solve the optimization problem with the pre-trained model"
    - Pre-trained model은 train 과정에서 얻을 수 있다.
- $\mathrm{Total\ cost} = 700\cdot C_{LA\to NY}$ ($C$: Cost)
- $\mathrm{Revenue} = D_{LA\to NY}\cdot m(700) $ ($D$: Demand, $m$: Predicted market share)
- $\mathrm{Profit=Revenue-Total\ cost}$
  - 이 중에서 $m(700)$만이 prediction. 이를 제외하고는 전부 fixed!
- ML Model은 주어진 feature를 이용해 이러한 predicted market share를 계산한다.
  - feature: ticket price, frequency, delay time, delay ratio, cancel ratio, stop, safety, craftsize, total seat
  - ![image-20200430164135561](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200430164135561.png) (Predicted market share of airline $i$)
  - ![image-20200430164153703](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200430164153703.png) (Passengers' <u>overall(not individual)</u> valuation about airline $i$)
    -   Linear combination of features($\eta$: learn해야 할 parameter, $X$: feature)
- 미국에는 현재 2000개 이상의 국내선 노선 존재($\to$2000개 이상의 regression model)
  - NY->LA, ... 등의 모든 노선이 prediction model이 되어 optimization module과 interact하는 방식
  - 각 노선은 서로 다른 coefficient를 가짐

- Game Theory - Prisoner's Dilemma
  - Nash equilibrium: 각 player가 자신의 전략을 바꿀 motivation이 없는 상태

  - Ex) Prisoner's dilemma

    ![image-20200430171534578](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200430171534578.png) 

    - 각 prisoner는 서로 소통할 수 없는 상황일 때, 이 경우 Nash equilibrium은 "모두 betray하는 경우" 이다.

  - 이러한 이론을 위의 airline 문제에 다시 적용해보면,

  - <u>4개의 major airline 사이에서 LA to NY 노선의 Equilibrium Ticket Price를 구하는 것!</u>

- 이렇게 equilibrium ticket price를 구했으면, dataset에 feature를 하나 추가한다.(Feature Engineering)

  - "Equilibrium ticket price - suggested price"
    - 양수일 경우: Ripping off(바가지)
    - 음수일 경우: market share increae
  - 이렇게 추가한 feature로 더 판단을 쉽게 할 수 있게 됨

### 정리

- Profit = Average tickt price * demand * $m_i(f_i)$ - $C_i$*$f_i$
- $\underset{<f_i>}{\max}\underset{R_i\in \mathcal{R}}\sum N_i(f_i)$를 구하는 문제
- 조건
  - $0\le f_i\le f_i^{max},\ f_i\in\mathbb{N},\ \forall R_i\in\mathcal{R}$
  - $\underset{R_i\in \mathcal{R}}\sum C_if_i\le b$

## Knapsack Problem

- 가장 유명한 decision making problem

- 많은 실생활의 문제들이 knapsack problem으로 모델링될 수 있다.
- "제한 된 크기의 knapsack(15kg)에 물건들(\$4-12kg, \$2-2kg, \$1-1kg, \$10-4kg, \$2: 1kg)을 넣으려 할 때, 어떻게 해야 가장 값어치가 높게 넣을 수 있을까?"
  - $\underset{x_i\in[0,1]}{\arg\max}\ \sum x_i\cdot v_i$, subject to $\sum x_ic_i\le b$
    - $v_i$: value of product $i$, $c_i$: cost(weight) of product $i$
  - 5개의 물건이 있다면 총 2^5^=32개의 가능한 경우 발생. O(2^5^)
    - 일부 경우는 문제 조건을 만족하지 못할 수도 있음
    - 가장 naive한 algorithm이나, product의 수가 많아질 수록 기하급수적으로 증가해 비효율적.
  - Hard constraint: $\sum x_ic_i\le b$
- 해결 방법 1. Greedy Algorithm
  - 매 iteration 마다 가장 비싼 product를 고르거나
  - efficiency=v/w or v/c 등으로 정의하고 가장 efficient한 product 순서대로 고른다!
  - 하지만 이러한 방법은 optimal solution을 제공하지 않는다... 더 나은 알고리즘이 존재!
  - product value가 계속 바뀌는 경우를 고려하지 못함

- 해결 방법 2. Dynamic Programming Algorithm
  - 특정 algorithm아니라, algorithm을 design하기 위한 paradigm에 가깝다!
  - $v[i,w]=max\{v[i-1,w], v[i-1,w-w[i]]+v[i]\}$ // 고르지 않음, 고름
  - ![image-20200430204220171](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200430204220171.png)  
    - 이렇게 계산해 나가다 보면 v[1,1]과 같은 overlapping node가 발생하는데, 이러한 경우는 한번의 계산으로 여러 번의 연산을 수행할 수 있기 때문에 시간이 절약되고, overlapping node가 많아질 수록 시간 절약이 더 많아진다.
      - Paradigm: Works well if many overlapping substructures in the recurrence tree
- Variations
  - 위의 경우는 고른다, 안고른다 의 문제이므로 0-1 Knapsack 이라고 부름
  - 이러한 변수를 최대 N개까지 고를 수 있는 문제의 경우 Integer Knapsack 이라고 부름!
    - Integer Knapsack의 경우 가짓수가 2^5^에서 5^5^로 늘어나듯이 더욱 complicated해진다.
    - Ex) $x_i\in\{0,1,2,\dots,N\}$: frequency in route $i$

### Airline Problem으로 적용

- Integer Knapsack
  - $v[i,j,w]$
    - $j$: choose integer number $j$ for route $i$
    - Ex) $v[2000,700,\$10B]$
      - 2000: route index number
      - 700: Maximum possible frequency in route 2000
      - $10B: budget
      - 여기서 시작해서 recurrence tree를 그려보면 첫 branch는 701개가 될 것이고, 각각 이어지면서 매주 큰 tree가 그려질 것이다.
      - branch는 각 profit이 더해지며 이어질텐데, 이러한 profit은 regression ML model에 의해 계산된다.
      - ![image-20200430210800430](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200430210800430.png) 
- $\underset{i}\sum v_ix_i$, where $x_i\in\{0,1,2,\dots\}$
  - $v_i$: <u>multi</u>-logit regression이므로
  - Non-linear Integer Knapsack Problem!

