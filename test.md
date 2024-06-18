# Regression Performance Metrics
>- 회귀 모델의 성능을 평가하는 데 일반적으로 사용되는 네 가지 지표는 평균 제곱 오차(MSE), 평균 제곱근 오차(RMSE), 평균 절대 오차(MAE) 및 결정 계수(R2)이다.



## Mean Squared Error (MSE)

**Definition**:  
평균 제곱 오차(MSE)는 예측 값과 실제 값 간의 제곱 차이의 평균이다.
제곱 과정으로 인해 더 큰 오류에 더 큰 비중을 둔다.

**Formula**:  
$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2$

**Interpretation**:  
- 낮은 MSE는 더 나은 적합성과 높은 예측 정확성을 나타냅니다.
- MSE는 큰 오류가 제곱되기 때문에 이상값에 민감합니다.

**Use Case**:  
MSE는 큰 예측 오류를 더 크게 벌주고자 할 때 유용하며, 큰 예측 오류가 특히 바람직하지 않은 응용 분야에 적합합니다.

---
# 4페이지
원문:

---

### Regression Performance Metrics

2. Root Mean Square Error (RMSE)

**Definition**:  
Root Mean Square Error (RMSE) is the square root of the average of the squared differences between the predicted values and the actual values. It provides a measure of the average magnitude of the prediction errors, expressed in the same units as the dependent variable.

**Formula**:  
$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2 }$

**Interpretation**:  
- RMSE is directly interpretable in the same units as the dependent variable.
- A lower RMSE indicates a better fit and higher prediction accuracy.
- Like MSE, RMSE is sensitive to outliers.

**Use Case**:  
RMSE is commonly used when you want an error metric that is easy to interpret and relates directly to the scale of the dependent variable.

---

번역:

---

### 회귀 성능 지표

2. 평균 제곱근 오차(RMSE)

**정의**:  
평균 제곱근 오차(RMSE)는 예측 값과 실제 값 간의 제곱 차이의 평균의 제곱근입니다. 이는 종속 변수와 동일한 단위로 표현된 예측 오류의 평균 크기를 제공합니다.

**공식**:  
$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2 }$

**해석**:  
- RMSE는 종속 변수와 동일한 단위로 직접 해석할 수 있습니다.
- 낮은 RMSE는 더 나은 적합성과 높은 예측 정확성을 나타냅니다.
- MSE와 마찬가지로 RMSE는 이상값에 민감합니다.

**사용 사례**:  
RMSE는 해석하기 쉽고 종속 변수의 규모와 직접적으로 관련된 오류 지표가 필요할 때 일반적으로 사용됩니다.

---
# 5페이지
원문:

---

### Regression Performance Metrics

3. Mean Absolute Error (MAE)

**Definition**:  
Mean Absolute Error (MAE) is the average of the absolute differences between the predicted values and the actual values. It provides a straightforward measure of the average prediction error magnitude, without emphasizing larger errors.

**Formula**:  
$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y_i} |$

**Interpretation**:  
- MAE is less sensitive to outliers compared to MSE and RMSE.
- A lower MAE indicates a better fit and higher prediction accuracy.

**Use Case**:  
MAE is useful when you need a simple, intuitive measure of prediction error that treats all errors equally, regardless of their magnitude.

---

번역:

---

### 회귀 성능 지표

3. 평균 절대 오차(MAE)

**정의**:  
평균 절대 오차(MAE)는 예측 값과 실제 값 간의 절대 차이의 평균입니다. 이는 더 큰 오류를 강조하지 않고 예측 오류의 평균 크기를 간단하게 측정합니다.

**공식**:  
$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y_i} |$

**해석**:  
- MAE는 MSE 및 RMSE에 비해 이상값에 덜 민감합니다.
- 낮은 MAE는 더 나은 적합성과 높은 예측 정확성을 나타냅니다.

**사용 사례**:  
MAE는 오류의 크기에 관계없이 모든 오류를 동일하게 취급하는 간단하고 직관적인 예측 오류 측정이 필요할 때 유용합니다.

---
# 6페이지
원문:

---

### Regression Performance Metrics

❖ Because the error value is squared,  
(1) when the error is between 0 and 1, the error is reflected smaller than the original,  
and (2) when the error is greater than 1, it is reflected larger than the original.

❖ All function values ​​are differentiable. Because MSE is a quadratic function, it does not have a peak.

---

번역:

---

### 회귀 성능 지표

❖ 오류 값이 제곱되기 때문에,  
(1) 오류가 0과 1 사이일 때는 오류가 원래 값보다 작게 반영되고,  
(2) 오류가 1보다 클 때는 원래 값보다 크게 반영됩니다.

❖ 모든 함수 값은 미분 가능합니다. MSE는 이차 함수이기 때문에 최고점을 가지지 않습니다.

---
# 7페이지
원문:

---

❖ Since the root is taken from MSE, the disadvantage of MSE in which errors are amplified is eliminated to some extent.  
❖ While MSE has an error function drawn in the form of a smooth curve, RMSE has a point that is not differentiable because it takes the root at the MSE.

---

번역:

---

❖ MSE에서 제곱근을 취하기 때문에 오류가 증폭되는 MSE의 단점이 어느 정도 제거됩니다.  
❖ MSE는 부드러운 곡선 형태로 오류 함수를 가지지만, RMSE는 MSE에서 제곱근을 취하기 때문에 미분 불가능한 점을 가집니다.

---
# 8페이지
원문:

---

❖ Because the values are not squared, it is robust (insensitive) to outliers.  
❖ There is a point in the function value that is not differentiable. Differentiation is impossible because the minimum value of the function expressing MAE is the vertex.  
❖ Unlike MSE and RMSE, equal weight is given to all errors.

---

번역:

---

❖ 값이 제곱되지 않기 때문에 이상값에 대해 강건(민감하지 않음)합니다.  
❖ MAE를 나타내는 함수의 최소값이 꼭지점이기 때문에 함수 값 중 미분할 수 없는 점이 있습니다.  
❖ MSE와 RMSE와 달리 모든 오류에 동일한 가중치를 부여합니다.

---
# 9페이지
![](https://velog.velcdn.com/images/kms39273/post/db49d008-3983-4f6a-9999-895157267fc5/image.png)

# 10페이지
원문:

---

### Regression Performance Metrics

4. Coefficient of Determination (R²)

**Definition**:  
The Coefficient of Determination (R²) measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides an indication of how well the independent variables explain the variability of the dependent variable.

**Formula**:  
$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$

where:  
- $SS_{\text{res}}$ is the sum of squares of the residuals,
- $SS_{\text{tot}}$ is the total sum of squares.

**Interpretation**:  
- R² ranges from 0 to 1.
- $R^2 = 0$ indicates that the model explains none of the variance in the dependent variable.
- $R^2 = 1$ indicates that the model explains all the variance in the dependent variable.
- Higher R² values indicate better model performance and explanatory power.

**Use Case**:  
R² is useful for understanding the overall fit of the model and comparing the explanatory power of different models.

---

번역:

---

### 회귀 성능 지표

4. 결정 계수 (R²)

**정의**:  
결정 계수 (R²)는 독립 변수로부터 종속 변수의 분산 중 예측 가능한 비율을 측정합니다. 이는 독립 변수가 종속 변수의 변동성을 얼마나 잘 설명하는지에 대한 지표를 제공합니다.

**공식**:  
$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$

여기서:  
- $SS_{\text{res}}$는 잔차 제곱합,
- $SS_{\text{tot}}$는 총 제곱합입니다.

**해석**:  
- R²는 0에서 1까지의 값을 가집니다.
- $R^2 = 0$은 모델이 종속 변수의 분산을 전혀 설명하지 못함을 나타냅니다.
- $R^2 = 1$은 모델이 종속 변수의 모든 분산을 설명함을 나타냅니다.
- 높은 R² 값은 더 나은 모델 성능과 설명력을 나타냅니다.

**사용 사례**:  
R²는 모델의 전반적인 적합성을 이해하고 다양한 모델의 설명력을 비교하는 데 유용합니다.

---
# 11페이지
원문:

---

Multiple Coefficient of Determination

- Relationship Among SST, SSR, SSE

$SST = SSR + SSE$
$\sum (y_i - \bar{y})^2 = \sum (\hat{y_i} - \bar{y})^2 + \sum (y_i - \hat{y_i})^2$

where:  
SST = total sum of squares  
SSR = sum of squares due to regression  
SSE = sum of squares due to error  

---

번역:

---

다중 결정 계수

- SST, SSR, SSE 간의 관계

$SST = SSR + SSE$
$\sum (y_i - \bar{y})^2 = \sum (\hat{y_i} - \bar{y})^2 + \sum (y_i - \hat{y_i})^2$

정의:  
SST = 총 제곱합  
SSR = 회귀에 의한 제곱합  
SSE = 오류에 의한 제곱합  

---
원문:

---

R² is the proportion of the total variation (SST) accounted for by the regression line (SSR).
$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$
$(0 \leq R^2 \leq 1)$

The closer it is to 1, the higher the explanatory power of the regression line.

---

번역:

---

R²는 회귀선(SSR)이 설명하는 총 변동(SST)의 비율입니다.
$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$
$(0 \leq R^2 \leq 1)$

값이 1에 가까울수록 회귀선의 설명력이 높습니다.

---

![](https://velog.velcdn.com/images/kms39273/post/bed64653-fa8e-4ded-b44e-8741b4ae080f/image.png)

# 12페이지
원문:

---

### Regression Performance Metrics

◆ The adjusted coefficient of determination is a metric designed to address the issue of the coefficient of determination increasing as explanatory variables are added. It achieves this by imposing a penalty on model complexity, thereby preventing overfitting, where the model becomes too tailored to the training data and loses predictive power on new data.  
◆ It adjusts the determination coefficient by utilizing the ratio of SSE (Sum of Squared Errors) to SST (Total Sum of Squares). Here, n represents the sample size, and k denotes the number of explanatory variables. This ratio is adjusted to account for degrees of freedom, enabling a more precise assessment of whether the additional explanatory variables yield significant improvements.  
◆ The formula for adjusted coefficient of determination is as follows:  
$R^2_{adj} = 1 - \left( \frac{SSE / (n - k - 1)}{SST / (n - 1)} \right)$

where,  
- $R^2_{adj}$ represents the adjusted coefficient of determination.
- SSE stands for Residual Sum of Squares.
- SST stands for Total Sum of Squares.
- n denotes the sample size.
- k represents the number of explanatory variables.

---

번역:

---

### 회귀 성능 지표

◆ 조정 결정 계수는 설명 변수가 추가됨에 따라 결정 계수가 증가하는 문제를 해결하기 위해 설계된 지표입니다. 이는 모델 복잡성에 대한 패널티를 부과하여 모델이 훈련 데이터에 과도하게 맞춰져 새로운 데이터에 대한 예측 능력을 잃는 과적합을 방지합니다.  
◆ 조정 결정 계수는 SSE(잔차 제곱합)와 SST(총 제곱합)의 비율을 활용하여 조정됩니다. 여기서 n은 샘플 크기, k는 설명 변수의 수를 나타냅니다. 이 비율은 자유도를 고려하여 조정되어 추가적인 설명 변수가 유의미한 개선을 가져오는지 더 정확하게 평가할 수 있게 합니다.  
◆ 조정 결정 계수의 공식은 다음과 같습니다:  
$R^2_{adj} = 1 - \left( \frac{SSE / (n - k - 1)}{SST / (n - 1)} \right)$

정의:  
- $R^2_{adj}$는 조정 결정 계수를 나타냅니다.
- SSE는 잔차 제곱합을 의미합니다.
- SST는 총 제곱합을 의미합니다.
- n은 샘플 크기를 나타냅니다.
- k는 설명 변수의 수를 나타냅니다.

---
# 13페이지
원문:

---

### Regression Performance Metrics

◆ The adjustment coefficient of determination does not necessarily increase when explanatory variables are added.  
◆ The adjusted coefficient of determination ranges between 0 and 1, with higher values indicating a better model fit to the data. However, a high value does not necessarily signify a good model. If the model is overly complex, it may suffer from overfitting issues. Therefore, when evaluating a model, it's crucial to consider not only the adjusted coefficient of determination but also other evaluation metrics.

---

번역:

---

### 회귀 성능 지표

◆ 조정 결정 계수는 설명 변수가 추가될 때 반드시 증가하지는 않습니다.  
◆ 조정 결정 계수는 0과 1 사이의 값을 가지며, 값이 높을수록 데이터에 대한 모델 적합도가 더 좋음을 나타냅니다. 그러나 높은 값이 반드시 좋은 모델을 의미하는 것은 아닙니다. 모델이 지나치게 복잡하면 과적합 문제가 발생할 수 있습니다. 따라서 모델을 평가할 때는 조정 결정 계수뿐만 아니라 다른 평가 지표도 고려하는 것이 중요합니다.

---
# 14페이지
# 15페이지
# 16페이지
# 17페이지
# 18페이지
# 19페이지