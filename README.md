# Monte Carlo Simulation of Stocks Using Brownian Motion

<!-- Table of Contents -->
# Table of Contents
- [About the Project](#about-the-project)
  * [Description](#description)
  * [Improvments](#improvments)

- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Run Locally](#run-locally)

- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
  

<!-- About the Project -->
## About the Project

This is a project I made in the course Inference Theory I at Uppsala University in Sweden. The full project can be seen in the pdf file "Full Project.pdf"

<!-- Description -->
### Description

The project Monte Carlo simulates future stock prices using Geometric Brownian Motion to model the data. Data of different stocks have been downloaded from Yahoo Finance between the period of early 2017 to early 2022 on a daily basis. An example of stock data form Amazon can be seen below:

<div class="align-center"> 
  <img src="https://github.com/EdwardGlockner/Brownian-Motion-Monte-Carlo-Stocks/blob/main/Images/historical_data.png" width="410" height="350"/>
</div>

The stock data is divided into a training set from which we calculate the model parameters of Brownian Motion. The model is then Monte-Carlo simulated until early 2022, where thousands of different price paths are calculated. Below is the simulation of Amazon stock:

<div class="align-center"> 
  <img src="https://github.com/EdwardGlockner/Brownian-Motion-Monte-Carlo-Stocks/blob/main/Images/Monte%20Carlo%20simulation.png" width="410" height="350"/>
</div>

We can calculate a histogram of the final values of all the simulated paths:

<div class="align-center"> 
  <img src="https://github.com/EdwardGlockner/Brownian-Motion-Monte-Carlo-Stocks/blob/main/Images/histogram.png" width="410" height="350"/>
</div>

After this, a confidence interval of the final values is calculated, and we can then see if the correct value lies in the interval. For example, for Amazon we get:

Mean simulated value:  142.16413158510346 <br />
Correct final value:  154.919495 <br />
Interval (139.45056870478774, 144.87769446541918) <br />

### Improvments

Often times, the correct value with not lie in the confidence interval. One method to try to improve our method is to make the Monte-Carlo simulation dynamic, as we were trading in real time. For each new point we see, we can update our model parameters which should lead to a more accurate simulation. In the current version we don't update the model parameters when we see new data points, which leads to inaccuracy in the model.


<!-- Getting Started -->
## Getting Started

<!-- Prerequisites -->
### Prerequisites

This project uses a number of libraries:

```bash
 matplotlib 
 pandas
 seaborn 
 scipy 
 datetime 
 numpy
 
```
  
<!-- Run Locally -->
### Run Locally

Clone the project

```bash
  git clone https://github.com/EdwardGlockner/Brownian-Motion-Monte-Carlo-Stocks.git
```

Open up a terminal and install the requirments

```bash
  python3 install.py
```

Now you can run the project

```bash
  python3 main.py
```



<!-- Contact -->
## Contact

Edward Glöckner - [@linkedin](https://www.linkedin.com/in/edwardglockner/) - edward.glockner5@gmail.com

Project Link: [https://github.com/EdwardGlockner/Brownian-Motion-Monte-Carlo-Stocks](https://github.com/EdwardGlockner/Brownian-Motion-Monte-Carlo-Stocks)


<!-- Links -->
## Links

Here are some helpful links:

 - [Brownian Motion Monte Carlo Simulation](https://medium.com/analytics-vidhya/building-a-monte-carlo-method-stock-price-simulator-with-geometric-brownian-motion-and-bootstrap-e346ff464894)
 
 
