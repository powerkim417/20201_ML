# 7. Decision Tree

## 과제

- Will teach only the basic concept of Decision Tree
- Write a report filling out details of Decision Tree

## 개념

### Basics of Decision Tree

- Decision Tree: SVM과 더불어 가장 유명하고 자주 사용되는 classification algorithm

- Random Forest Classifier: Ensemble of many DTs
  
  - 주어진 test case $x$에 대해, 각 Decision Tree(DT~1~, DT~2~, .., DT~n~)는 각각 조금씩 다른 결과를 predict할 것이다.
  - RFC는 이 결과들을 모두 합(aggregate)하여 하나의 reliable한 ensemble of all those predictions를 낼 것이다.
  - 높은 성능!
  
- Ex) 사람이 긴 머리를 가졌는지 아닌지 classify하려 할 때

  - $f(x)=0$(short hair), $f(x)=1$(long hair)

  - where $x$ is a vector of features describing each individual

    - $x$=[gender, height, weight, age, ...] 와 같이 둘 수 있음

  - Quality measure for each feature

    - target feature, target label, data 필요
      - label: long or short hair
      - data의 중요성
        - M(gender, label, data)와 같은 경우 일반적으로는 gender에 따라 머리 길이를 유추하기 쉽지만, 모든 남자가 머리를 기르는 지역의 data는 유추하기 어려워진다.

  - 과정

    1. Evaluate the quality measure for each feature.
    2. Choose the best feature/split criteria to build a note.
       - Gender로 나누었을 때 male의 99%가 short hair, female의 99%가 long hair로 나왔다면, 1번의 split으로 reliable한 DT를 만들 수 있다.
       - 이 때, 이 99%의 short hair에서도 split(long, short)을 할 수 있음

    1,2의 반복(iterated until a certain stop criteria(=max depth) is met)

### 과제 설명

- Decision tree construction algorithm의 구성
  1. Tree growing
     
     - Classification performance를 향상시키기 위해 노드를 하나씩 추가하는 단계
     
     - ```pseudocode
       TreeGrowing (S,A,y)
       Where:
       S - Training Set
       A - Input Feature Set
       y - Target Feature
       Create a new tree T with a single root node.
       IF One of the Stopping Criteria is fulfilled THEN
           Mark the root node in T as a leaf with the most
           common value of y in S as a label.
       ELSE
           Find a discrete function f(A) of the input attributes values such that splitting S according to f(A)’s outcomes (v1,...,vn) gains the best splitting metric.
           IF best splitting metric > threshold THEN
               Label t with f(A)
       	    FOR each outcome vi of f(A):
           	Set Subtreei= TreeGrowing (σ{f(A)=vi}S,A,y).
           	Connect the root node of tT to Subtree_i with an edge that is labelled as vi
           	END FOR
           ELSE
               Mark the root node in T as a leaf with the most common value of y in S as a label.
           END IF
       END IF
       RETURN T
       ```
     
     - 대부분의 discrete splitting function은 univariate(하나의 attribute에 의해 노드가 split됨)
     
     - 가장 적절한 attribute를 찾아야함
     
     - criteria
     
       - origin of the measure에 의해: information theory, dependence, distance
       - measure structure에 의해: impurity based criteria, ...
       - impurity-base criteria
         - k개의 값을 가진 랜덤변수 x와 확률 P
         - 
  2. Node pruning
     
     - Overfitting을 막기 위해 노드를 하나씩 지우는 단계
     
     - ```pseudocode
       TreePruning (S,T,y)
       Where:
       S - Training Set
       y - Target Feature
       T - The tree to be pruned
       DO
           Select a node t in T such that pruning it maximally improve some evaluation criteria
           IF t!=Ø THEN T=pruned(T,t)
       UNTIL t=Ø
       RETURN T
       ```
     
       