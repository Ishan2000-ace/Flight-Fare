# Flight-Fare-Prediction
## Overview of The Project
This projects simply takes passenger's details like date of journey,Route,No. of stops etc and predicts the fare of the journey
![Flight](https://www.nairaland.com/attachments/4783490_113_jpege7be8ad614b8bd81d7befac561bce03f)
## Dataset
I used kaggle dataset for this project. It can viewed and downloaded from this link:[Dataset](https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh/)
## Machine Learning Algorithm
Random Forest fits a number of decision tress of various sub sample of the datasets and average them to provide better accuracy and to avoid overfitting of the data. By defalut no. of trees that are combined are 100 but we can change them according to our needs.
More details can be viewed at [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=A%20random%20forest%20regressor.%20A%20random%20forest%20is,to%20improve%20the%20predictive%20accuracy%20and%20control%20over-fitting.) It is official website of sklearn
![Random Forest](https://camo.githubusercontent.com/fdb21b37f4e52818e7c4ac433ace0773599cdffc03ea468e362b5d00ea8ea1d4/68747470733a2f2f686172647461736b73696e2e66696c65732e776f726470726573732e636f6d2f323031392f30362f72616e646f6d2d65313536313732393938303831352e706e673f773d35323226683d333437)
![sklearn](https://camo.githubusercontent.com/a670f346bb5a8d656ee1255bd5325d3eb41a078b61f8a4799a293b27a2ded066/68747470733a2f2f6d616368696e652d6561726e696e672e6e65742f77702d636f6e74656e742f75706c6f6164732f323031382f30322f6579655f736b6c6561726e2e706e67)
## Deployment of the Model
This model is deployed by using flask framework of python that is used for backend development of web. HTML and CSS has been used for the frontend UI. This app can be viewed at
[Flight Fare Predictor](https://flightfareprediction.azurewebsites.net/).This app is being hosted on Azure cloud.
![Azure Cloud](https://abouttmc.com/wp-content/uploads/2019/02/logo_azure.png)
## Installation Details
Steps for Installation:
1. First we have to download Anaconda we can visit the page directly by clicking at [Anaconda Download](https://www.anaconda.com/products/individual)
Here we have scroll to the bottom and we can see installers for various opearating systems. We have to choose according to our system requirements.
This step can be skipped if anaconda is already installed.

2. In anaconda prompt create a new environment by the following command:
```conda create --name myenv```

3. After creating a new environment it is activated by following command: ```conda activate myenv'''

4. Now we have to navigate to directory where we have downloaded this repository and we have to it via anaconda prompt using command: ```cd path of the directory``` 

5. After navigating to the directory we have to install the dependencies of this project by using a file that is called requirements.txt that I have already provided in 
my repository and the dependencies can be installed by using command ```pip install requirements.txt```
but this command must be executed after executing step 4 otherwise it will won't work.

6. To run this app in the local machine we have to use the command:```python app.py```
After execution it will give a local address just copy that and paste that in the address bar of the browser and the web app is ready to use.
## Credits
Credits of this project goes to krish Naik sir's YouToube channel. Lead data scientist at Ineuron. His videos were a great help
[Link of the channel](https://www.youtube.com/user/krishnaik06)
