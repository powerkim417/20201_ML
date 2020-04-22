# 4-1. Basis Function and Non-linear Regression

## Linear Model

- 정의

  - True function $y$가 주어졌을 때, $y(x)=w_0+w_1x_1+...+w_Dx_D=w_0+\underset{j}\sum w_jx_j$
    - $x$는 vector이고, $w_i$는 parameter

- 그런데 만약 $y$가 linear하지 않다면? linear regression에 의한 위 approximation은 정확하지 않을 것이다.(not accurate)

  $\Rightarrow$ Non-linear regression의 필요성

  - Basis function 사용

## Non-linear Model with Basis Function

### Basis Function

- Basis vector와 유사한 개념

  - Basis vector: 주어진 vector $v=[a,b,c]$를 $a\cdot[1,0,0]+b\cdot[0,1,0]+c\cdot[0,0,1]$와 같은 linear combination으로 나타낼 때, $[1,0,0], [0,1,0], [0,0,1]$이  basis vector가 된다.

- 이와 같은 concept은 mixture of gaussian, fourier series 등에서도 사용됨

  - Mixture of gaussian: any probability distribution can be approximated by linear combination of gaussian distributions
  - Fourier series: any periodic function can be approximated by infinite number of sine functions at worst case

- $y(\mathbf{x,w})=w_0+\underset{j=1}{\overset{M-1}\sum}w_j\phi_j(x)$로 나타낼 때, $\phi$가 basis function

  - 총 M개의 parameter(feature는 M-1개)
  - $y$를 basis function의 linear combination으로 나타냄
  - 이러한 형태로 non-linear function을 approximate할 수 있음
  - True function이 매우 복잡한 경우, $M$값이 매우 커질 수 있음

- Basis function은 다양한 형태로 존재 가능! (e.g. sine function, RBF, ...)

  - RBF(Radial Basis Function)

    - $\phi_j(x)=\mathrm{exp}(-\dfrac{(x-\mu_j)^\top(x-\mu_j)}{2\sigma^2_j})$

      <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200410205025834.png" alt="image-20200410205025834" style="zoom: 80%;" />

    - $x$의 값이 mean value에 근접할 때, 함수값이 최대(Gaussian function의 형태와 유사)

    - 정확한 표기는 $\phi_j(x;\mu_j,\sigma_j)$

      - $x$(input vector) parameterized by $\mu_j$(mean vector), $\sigma_j$(standard deviation of RBF)

    - $(x-\mu_j)^\top(x-\mu_j)=||x-\mu_j||_2^2$ , 즉 Squared L2-norm으로 scalar 값이 된다.

      - 따라서 $\phi_j(x)$의 값도 모두 scalar로 나온다.

    - 다양한 $\mu$와 $\sigma$로 RBF를 사용하면 어떤 non-linear function도 approximate할 수 있다.

### Non-linear model regression

- Non-linear regression model: $y(\mathbf{x})=w_0+\underset{j=1}{\overset{M-1}\sum}w_j\phi_j(x;\mu_j,\sigma_j)$
  
- Learn해야 할 parameter: $w_0, w_j, \mu_j, \sigma_j(1\le j\le M-1)$
  
- 한편, Linear regression의 경우 optimal parameter에 대한 analytical solution(일반해?)을 calculate할 수 있었다.

- 하지만, Non-linear regression model의 경우 Basis function의 존재로 인해 analytical solution을 구할 수 없다.

  $\Rightarrow$ "Gradient descent" method를 활용해야 한다.

  - Machine learning field에서 가장 간단한 optimization algorithm

#### How to train

$y(\mathbf{x})=w_0+\underset{j=1}{\overset{M-1}\sum}w_j\phi_j(x;\mu_j,\sigma_j)$

Learn해야 할 parameter: $w_0, w_j, \mu_j, \sigma_j(1\le j\le M-1)$

![image-20200410215155411](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200410215155411.png)

- 실제로는 아래 그림처럼 bias가 아래에 존재해야 한다...

  ![image-20200411012625248](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411012625248.png)

##### 그림 예시(Neural network-like representation of RBF-based non-linear regression, RBF Network)를 통한 설명

<span style="color:red">ground truth를 hat으로 설명하던데, 잘못설명한건지 확인 필요</span>

- 3-dimension의 vector $x$, 4개의 RBF(각각 다른 $\mu$ 및 $\sigma$), 2-dimension의 predicted output vector $\hat b$, ground truth에 의한 output vector $b$
- vector를 각 RBF에 넣었을 때 서로 다른 scalar value가 나온다.
- $\hat{b}$의 각 element는 RBF의 결과로 나오는 4개의 scalar value에 대한 weighted combination을 값으로 가진다.
  - $\hat{b}_0=W_{00}\cdot\mathrm{scalar_0}+W_{10}\cdot\mathrm{scalar_1}+..$
- (Define a loss function) 실제 output 결과 $b$(ground truth)와, predicted output $\hat b$에 대해, 
  - $e_j=b_j-\hat{b}_j$
  - $E=\dfrac{1}{2}(e_0^2+e_1^2)=\dfrac{1}{k}\underset{k}\sum {e_k}^2$ (MSE)
- 이제 그림 내의 $\mu_j, \sigma_j, W_{j,k}$에 대해 learn해야 한다!
- (Back propagation)
  - Gradient Descent
    - $W_{j,k}=W_{j,k}-\lambda\cdot\Delta W_{j,k}$ ($j$: input vector의 element #, $k$: output vector의 element #)
  - $\Delta W_{j,k}\equiv\dfrac{\partial E}{\partial W_{j,k}}$
    - The change rate of $E$ given a small change on $W_{j,k}$
    - $W_{j,k}$를 조금 변화시킬 때 이 변화가 $\hat{b}_k, e_k, E$에 순차적으로 미친다.
      - 이 아이디어는 아래 chain rule에 그대로 적용됨
  - $\Delta W_{j,k}\equiv\dfrac{\partial E}{\partial W_{j,k}}$
    $=\dfrac{\partial E}{\partial e_k}\dfrac{\partial e_k}{\partial W_{j,k}}$ (by Chain rule)
    $=e_k\dfrac{\partial e_k}{\partial W_{j,k}}$ (by Partial derivation, $\dfrac{\partial E}{\partial e_k}=e_k$)
    $=e_k\dfrac{\partial e_k}{\partial \hat{b}_k}\dfrac{\partial \hat{b}_k}{\partial W_{j,k}}$ (by Chain rule)
    $=-e_k\dfrac{\partial \hat{b}_k}{\partial W_{j,k}}$ (by Partial derivation, $\dfrac{\partial e_k}{\partial \hat{b}_k}=-1$)
    $=-e_k\phi_j$ (by Partial derivation, $\hat{b}_k=...+W_{j,k}\cdot\mathrm{scalar_j}+...$ 꼴이므로 $\dfrac{\partial \hat{b}_k}{\partial W_{j,k}}=\mathrm{scalar}_j=\phi_j$)
    $\therefore \Delta W_{j,k}=-e_k\phi_j$
  - 이를 위 식에 다시 적용하면 ==$W_{j,k}=W_{j,k}+\lambda\cdot e_k\phi_j$==가 된다.
    - 사실 올바른 표기는 ${W_{j,k}}^{(k+1)}={W_{j,k}}^{(k)}+\lambda\cdot e_k\phi_j$ 인데,
    - 이 superscript는 iteration number를 의미한다.

# 4-2. Gradient Descent

The simplest optimization algorithm

Widely-used to train machine learning model(e.g. Deep Neural Networks)

##### 그림을 통한 개념 설명

<img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411013406130.png" alt="image-20200411013406130" style="zoom:80%;" />

- $\lambda$는 step size(보폭)이고, 화살표들은 가는 방향으로의 vector
- $x_t$가 가장 꼭대기 position
- Gradient-based optimization은 위 그림처럼 산을 오르고 내리는 것과 관계있다.
- 이에 의하면 Gradient descent method는 objective function이 <u>unconstrained convex</u>(단순 아래로 볼록)할 때 optimal solution을 찾을 수 있다.
  - Gradient를 따라 조금씩 내려가면 global minimum을 찾을 수 있음

### Gradient Descent

- $f$ is convex with respect to $\theta$일 때 gradient descent method를 사용해 $\underset{\theta}{\mathrm{argmin}}f(\theta)$를 solve할 수 있다.

  - 만약 $f$가 convex하지 않다면, final situation이 initial point에 따라 결정된다.

  <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411014338683.png" alt="image-20200411014338683" style="zoom:50%;" /> (Irregular surface)

  - 이 경우는 gradient descent가 잘 작동하지 않는다.
    - starting point에 따른 local optimum만을 구할 수 있음

- $\theta^{(k)}=\theta^{(k-1)}-\lambda\nabla f$ (the gradient of $f$ with respect of $\theta$)

  - $\nabla f=\nabla\theta=\nabla_\theta f$
  - minimization problem을 해결하기 위한 gradient descent vector
  - iterate 과정을 여러번 반복하면 optimal position을 얻을 수 있음

- Taylor Expansion

  - $f(x)=f(a)+\dfrac{f'(a)}{1!}(x-a)+\dfrac{f''(a)}{2!}(x-a)^2+\dfrac{f'''(a)}{3!}(x-a)^3+...$

    - 단, 이 식은 x=a 근처에서만 accurate하다.
    - 전체 구간에서 accurate하려면, 항의 갯수가 무한해야 하는데, 그럴 수 없으므로..

  - 일반적으로 1st/2nd order derivative(1차항/2차항까지만을 나열한 식)만을 사용한다.

    - Ex) 2nd order Taylor expansion: $f(x)=f(a)+\dfrac{f'(a)}{1!}(x-a)+\dfrac{f''(a)}{2!}(x-a)^2$

      <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411031659512.png" alt="image-20200411031659512" style="zoom:80%;" />

  - 3rd order 이상의 식을 사용하면 위의 그림보다는 더 accurate할 것이다.

  - <u>Gradient Descent의 경우 2nd order Taylor expansion과 밀접한 관련이 있다.</u>

- Finding $\theta^*$

  - 2nd order Taylor expansion 식을 vector에 대해 확장하면 다음과 같다.
  - $f(\theta)=f(a)+\nabla f(a)^\top(\theta-a)+\dfrac{1}{2}\nabla^2f(a)||\theta-a||_2^2$
    - This 2nd-order Taylor expansion is accurate around $\theta=a$
  - $\nabla^2f(x)=\dfrac{1}{t}I=\begin{pmatrix} \dfrac{1}{t} & 0 & \cdots & 0 \\ 0 & \dfrac{1}{t} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \dfrac{1}{t}\end{pmatrix}$ 이라고 가정할 때,
  - $f(\theta)=f(a)+\nabla f(a)^\top(\theta-a)+\dfrac{1}{2t}||\theta-a||_2^2$
    - $\dfrac{1}{2t}||\theta-a||_2^2$: Proximal term ($\theta$ should be close to $a$)
  - <span style="color:red">여기서부터 이해 하나도 안됨</span>
  - 위 식에서 $f(x)$를 $f'(x)$로 바꾸면, $f'(\theta)=f(a)+\nabla f(a)^\top(\theta-a)+\dfrac{1}{2t}||\theta-a||_2^2$
    - $f'$는 미분이 아니라 $f$와 구분짓기 위한거..
  - 그러면 $\underset{\theta}{\mathrm{argmin}}f'(\theta)$을 구하기 위해 우항을 minimize하면
  - $\theta^*=a-t\nabla f(a)$ (taylor expansion을 최소로 하는 값, 즉 next position)
    - <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411034536034.png" alt="image-20200411034536034" style="zoom:80%;" />
  - Gradient Descent는 2nd-order taylor expansion-based method의 special case이다.
    - $\nabla^2f(x)$를 계산하지 않고 assume해야 하므로<span style="color:red">??</span>
    - 따라서, Gradient descent method는 finding minimization position of 2nd-order taylor expansion approximation와 같다.<span style="color:red">??</span>

### Geometrical Interpretation

##### 예시

<img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411035107753.png" alt="image-20200411035107753" style="zoom:67%;" />

- 위와 같은 objective function이 있다고 가정

- 이의 minimum point는 바닥 꼭지점.

- 이 objective function의 surface를 위에서 아래로 내려다보면

  <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411035215139.png" alt="image-20200411035215139" style="zoom: 50%;" />

- 이렇게 보일 것이고, 층에 따라 각각 다른 circle이 그려질 것이다.

- 이러한 circle 하나를 따라갈 때, objective value(여기서는 높이)는 모두 같은 값이다.

  - 1개의 circle은 1개의 level set을 만들 것이다.

#### Level Set

- 정의: objective value가 같은 모든 점들의 집합

#### Gradient

- <u>Increasing direction of objective function</u>

  <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411035215139.png" alt="image-20200411035215139" style="zoom: 50%;" />

- Gradients are perpendicular to tangent lines(접선) to the level set.

  - Ex)  ![image-20200411155639961](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411155639961.png)와 같은 경우
    - $f(x,y)=x^2+y^2$
    - $\nabla f(x,y)=\begin{bmatrix}\dfrac{\partial}{\partial x}(x^2+y^2)\\\dfrac{\partial}{\partial y}(x^2+y^2)\end{bmatrix}=\begin{bmatrix}2x\\2y\end{bmatrix}$
    - 그림에서 빨간 점의 경우 $x=0, y<0$이므로, $\nabla f(x,y)=\begin{bmatrix}0\\2y\end{bmatrix}$가 되어 gradient는 아래로 향하는 vector가 되고, line tangent와는 수직이 된다.
      - 이는 다른 모든 점에 대해서도 성립한다.

### Gradient Descent의 문제점

- 특정 상황에서 잘 작동하지 않음(e.g. Zig-zag problem)

  <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200411162944976.png" alt="image-20200411162944976" style="zoom:80%;" />

  - 이상적인 trajectory optimization은 오른쪽 그림처럼 straight하게 내려가야 하지만, gradient descent를 이용하면 왼쪽 그림처럼 zig-zag 모양으로 trajectory optimization이 진행된다.

- 이러한 zig-zag problem을 해결하기 위해서는 2차항을 $\dfrac{1}{t}I$ 이라고 예측하지 않고, "real" 2nd order derivative method를 사용해야 한다. (e.g. L-BFGS)

  - 이 방법에서는 Hessian Matrix를 계산해야 하는데, quadratic workload가 소요된다.

- Hessian Matrix

  - $H_{i,j}=\dfrac{\partial^2f}{\partial x_i \partial x_j}$
  - n개의 parameter를 learn해야 할 때 $H$의 size는 n\*n\*d(vector dimension)
    - 이 때, $n^2$ term이 나오므로 일반적으로 계산하기 힘들어서 prohibited.
    - 이러한 이유로 인해 결국 gradient descent에 의존하게 된다.

##### 결론

- Gradient descent는 zig-zag problem이라는 문제가 있지만, 이를 해결하기 위한  "real" 2nd order derivative method는 일반적으로 계산하기 힘들어서 결국 gradient descent를 사용해야 함
