# Econ445 Group project
# Rivalry-Beyond-Campus:Identifying-Real-Estate-Opportunities-for-School-Districts-using-Machine-Learning

## Table of Content
- [Abstract](#abstract)
- [Task Description](#task-description)
- [Major Challenges and Solutions](#major-challenges-and-solutions)
- [Data Description](#data-description)
- [Modelling](#modelling)
- [Summary](#summary)

## ABSTRACT
In view of the increased uncertainty in the real estate market over the pandemic of COVID-19, it is more important for investors and customers to avoid uninformed decisions. In this project, we built a machine learning framework to estimate expected value of residential properties in the university district of both UCLA and USC. We developed a set of regressors over tax assessment data and Zillow housing data and identified 5 investment opportunities for each school district. We also explained such results based on a naïve value discovery mechanism. The Stacked Ensemble model and SVR model in our system showed best out-of-sample performance in terms of MAE and MSE respectively. The district of UCLA is a better submarket in term of its higher opportunity density.

<div align=center><img alt="UCLA and USC Community" src="https://raw.githubusercontent.com/ShuangfeiLi/Identifying-Real-Estate-Opportunities-for-School-Districts-using-Machine-Learning/main/img_folder/uclausc.png" />></div>
<p align="center">Fig.1. UCLA and USC Community</p>

## TASK DESCRIPTION
1. Explore the major drivers for the value of properties.
2. Identify at least 5 investment opportunities of residential properties for both UCLA district and USC district, and answer which school district is the winner overall.

## MAJOR CHALLENGES AND SOLUTIONS
Based on the structural nature and expected functions of our predictive framework, there are at least two major challenges that should be understood properly: the determination of opportunity discovery mechanism and the data proxy issue.

### A. Opportunity Discovery Mechanism
We believe that the price of a property would deviate from its expected value due to some uncaptured information and unpredictable factors. We use 𝑝𝑖 and 𝑝̅𝑖 to denote the market price and expected value of property 𝑖. The property would be identified as an investment opportunity when 𝑝𝑖<𝑝̅𝑖, because it could generate immediate profit if sold right after its purchase, assuming no significant transaction costs. Then, we would be able to train a machine learning model 𝑓: ℝ𝑑→𝑝̂𝑖 over 𝑑 dimensional data to estimate 𝑝̅𝑖 for each in-sample property. And if 𝑝𝑖<𝑝̂𝑖 , the model predicts property 𝑖 is an opportunity.

### B. Proxy Selection
Based on the opportunity discovery model proposed above, the next challenge is collecting relevant data to make model estimations reasonable. As we know, for property 𝑖, its expected value 𝑝̅𝑖 is always unrevealed. Feeding the model with proper proxies is important to the model success. Also, although market price 𝑝𝑖 is always available on paper, the majority of records are associated with closed deals. So, these backward-looking signals would introduce the risk of bad generalization, thus making our application unpractical.

To address these problems, we choose the tax-based assessor parcels data as our major dataset to estimate the expected value of properties. We believe it would contribute as a good proxy because unreasonable evaluations directly lead to tax loss or crisis of social justice. Besides, we turn to Zillow data to make sense of the market as well. To be specific, we use prices of historical deals to adjust the estimation of 𝑝̅𝑖, and choose real-time listing prices of properties for sale as the proxy of 𝑝𝑖 to make our predictions forward-looking.

## DATA DESCRIPTION
The datasets used in this project can be roughly divided into three categories: **location data, property characteristics data, and price data**. The location data and property characteristics data are obtained from [Assessor Parcels Data](https://dev.km2.ai/public/parcels_last.csv) as part of County of Los Angeles open data, the price data is collected from the same dataset and extra [Zillow housing](https://www.zillow.com/?utm_content=1471764169|65545421228|kwd-570802407|509015461845|&semQue=null&k_clickid=_kenshoo_clickid_&gclid=Cj0KCQiA95aRBhCsARIsAC2xvfx3PwnD_833nEwyoiwa0D3eCoUd0BemV-Om8vHDwFLjjDIvZy6aIGQaAgY4EALw_wcB) data.

The raw datasets include 38 million properties with descriptions for parcels on the assessor's annual secured assessment rolls from 2006 to 2021. In order to have a good consistency between datasets, we filtered properties by restricting their assessment roll year between 2013 and 2021. Selected features are summarized as Table I.

**Table I. SELECTED FEATURES AND TARGET**
| Name | Description  |
| :--: |    :-:   |
| **Category1** | **Location Features** |
| zip2 | The 5-digit zip code that matches property’s actual street address |
| distance_to_ucla | The miles distance away from UCLA  |
| distance_to_usc | The miles distance away from USC  |
| PropertyLocation | The actual address of the property (used to match real-time listing price)  |
| **Category2** | **Property Characteristics Features** |
| Bedrooms | The total number of bedrooms  |
| Bathroom_per_bedroom | The total number of bathrooms/ total number of bedrooms  |
| Units | The total number of living units  |
| YearBuilt | The year property was originally built  |
| Years_until_effective | The number of years between build year and effective year  |
| **Category3** | **Price Features/Target** |
| LandValue_percent | The proportion of a property’s land value to its total value  |
| Price_per_unit | The total value/ total number of living units  |
| ZHVI | Zillow Home Value Index, the typical (35th to 65th percentile range) home value of a zip code  |
| ZHVI_sf | The typical price per square foot for a property: Zillow Home Value Index / the total square footage  |
| price_sf | The price per square foot for a property (the target of models)  |
| … | …  |


## MODELLING
Relevant research has indicated several powerful machine learning algorithms widely used in regression problems of real estate pricing. Here we choose five of them to build a set of competing regressors in our system.

### K-Nearest Neighbors (KNN)
KNN is a non-parametric algorithm to perform regression or classification. This distance-based method always requires feature normalization or standardization. Given a positive integer K and one observation 𝑥𝑖, the algorithm will identify the K points that are closest to 𝑥𝑖. Then, in a regression setting, the predicted output can be generated by aggregating the selected K points, such as computing the average. The most important hyper-parameter of KNN is the value of integer K.

### Random Forest (RF)
Random forests operate by constructing a multitude of decision trees in the training process and outputting a prediction of individual trees based on the parameters inputted. These large number of decision trees are built over bootstrapped samples. If a simple decision tree model is trained on B number of bootstrap samples, then the prediction of the RF, denote as 𝑓𝑅𝐹, will be the average of individual predictions, denote as 𝑓∗, coming from these decision trees. That is:

<p align="center">𝑓𝑅𝐹̂(𝑥)= 1𝐵Σ𝑓∗𝑏(𝑥)𝐵𝑏=1 (2)</p>

In each split, a random set of features is selected as split candidates, but only one of these predictors would be used in the split. In this way, RF algorithm has the advantage of decorrelating predictors, thus reducing the variability of the model performance.

### Support Vector Regression (SVR)
Following the principle of the Support Vector Machine, SVR aims to find a function that presents a margin of tolerance from the target values. It maps training examples to points and tries to maximize the width of the gap between the predicted and true outputs. To address the more common situation that nonlinear relationships between target values, a special kernel approach will be applied to enlarger the feature space.

C could be one of the most important hyper-parameters when applying SVR. In general, when C is relatively small, the tolerance of violations to the margin would also be small. So, it always leads to a narrow margin. As the C goes up, we will tolerate more violations to the margin, thus leading to a wider margin.

### XGBoost (XGB)
XGB is a tree-based ensemble method optimized to perform efficient and flexible predictions within the framework of gradient boosting. Unlike the traditional tree building process, this method builds sequential trees using a parallelized implementation, allowing us to achieve relatively low prediction errors with modest memory and runtime requirements. Another advantage is its excellent performs over complex and high‐dimensional data.

### Stacked Ensembles (SE)
The thinking of Stacked Ensemble here is to use multiple learning algorithms (weak regressors) we built already to build a model with better predictive performance (strong regressor). Indeed, the RF algorithm is an ensemble-based leaner. The method of stacked ensemble tries to find the optimal combination of a collection of prediction algorithms using a process called Stacking or Super Learner. H2O ai has automated most of the steps in this algorithm, so now it is much easier to implement it in data science projects.

## SUMMARY

**1. Feature Importance**

Here we will use our XGB models to depict the global structure of predictors:

<div align=center><img src="https://raw.githubusercontent.com/ShuangfeiLi/Identifying-Real-Estate-Opportunities-for-School-Districts-using-Machine-Learning/main/img_folder/Feature%20importance%20of%20XGB%20model%20for%20UCLA.png" width="400px"/>></div>
<p align="center">Fig.2. Feature importance of XGB model for UCLA</p>

For the UCLA district, the land value proportion to the total value is the most important numerical drivers, distance to UCLA is the second one.

<div align=center><img src="https://raw.githubusercontent.com/ShuangfeiLi/Identifying-Real-Estate-Opportunities-for-School-Districts-using-Machine-Learning/main/img_folder/Feature%20importance%20of%20XGB%20model%20for%20USC.png" width="400px"/>></div>
<p align="center">Fig.3. Feature importance of XGB model for USC</p>

For the USC district, as shown in Fig.3, although land value proportion is relatively important as well, price per unite is the most important factor, distance to the campus is not as important as it for UCLA. For both districts, the typical market price at the time of assessment we introduced from Zillow is one of the top 10 drivers.


**2. Investment Opportunities Discovery**

Back to our second purpose, here we identified 5 investment opportunities for each university district based on the suggestion of our best SVR models. We adjusted the estimation of expected value we generated for each property and compare to Zillow's Best Estimate, which represents the market’s signal. If the expected value of one property higher than the market’s current pricing, it would be regarded as an undervalued property thus a potential opportunity.

TABLE II. OPPORTUNITIES DISCOVERY: MODEL EVIDENCE FOR UCLA ($/SQFT)   |  TABLE III. OPPORTUNITIES DISCOVERY: MODEL EVIDENCE FOR USC ($/SQFT)
:-------------------------:|:-------------------------:
![](https://github.com/ShuangfeiLi/Identifying-Real-Estate-Opportunities-for-School-Districts-using-Machine-Learning/blob/main/img_folder/MODEL%20EVIDENCE%20FOR%20UCLA.png)  |  ![](https://github.com/ShuangfeiLi/Identifying-Real-Estate-Opportunities-for-School-Districts-using-Machine-Learning/blob/main/img_folder/MODEL%20EVIDENCE%20FOR%20USC.png)

For the properties listed in Table II and Table III, we believe they are undervalued and have the potential of obtaining a higher price per square feet in the future.
