# Premier League Match Prediction Using Machine Learning and Probabilistic Models
This project aims to evaluate the predictive accuracy of different machine learning and probabilistic models when it comes to predicting English Premier league football.

## Overview
This project uses a three-phase development process each with its own prediction and evaluation stage. The phases are as follows:

### Phase 1 - Machine Learning
Phase 1 uses base match statistics to see if Logistic Regression, Random Forest and Decision Tree models can accurately predict matches and seasons.

### Phase 2 - Poisson Distribution
Phase 2 uses expected goals and different types of team strength calculations to evaluate the predictive accuracy of the Poisson distribution.

### Phase 3 - Advanced Distributions
Phase 3 buils on Phase 2 by implementing the Laplace, Negative Binomial and Bayesian Poisson models to evaluate if relaxing Poisson assumptions improves predictive accuracy.

## Project Structure

- Phase 1/
  - data/
    - processed/
    - raw/
  - src/
- Phase 2/
  - Data/
  - scripts/
- Phase 3/
  - Data/
  - scripts/
- README.md
- requirements.txt
- .gitignore

### Data source Declaration
The data for Phase 1 has been obtained from [football-data](https://www.football-data.co.uk/data.php). The workbooks used are store in /Phase 1/Data/Raw.
The data for Phase 2 and 3 has been obtained from [understat](https://understat.com/). The datasets used cannot be stored on github, there is a python script included which can download the datasets locally. **Only run this local to device, not on github**.

## Installation Requirements
- python version 3.13.2 or later
- aiohttp 3.13.3
- matplotlib 3.10.8
- nest_asyncio 1.6.0
- numpy 2.4.4
- pandas 3.0.2
- scikit_learn 1.8.0
- scipy 1.17.1
- understat 0.1.14

These can be installed using the requirements.txt file through the following terminal code: `pip install -r requirements.txt`

## Running the project
For running this project, it must be run **in a local environment, in order to see graphs it must be ran in an IDE**.
Each Phase has its own main.py file, so they should be run separately. It is important to note that before running the terminal code for each phase below, **we must cd into the root directory**. Run the project through the terminal as follows:

### Phase 1 
- `cd 'Phase 1'`
- `cd src`
- `python main.py`
This runs all scripts in order, and generates the evaluation tables and graphs.
### Phase 2
- `cd 'Phase 2'`
- `cd scripts`
- `python main.py`
This runs all scripts in order, and generates the evaluation tables and graphs.
### Phase 3
- `cd 'Phase 3'`
- `cd scripts`
- `python main.py`
This runs all scripts in order, and generates the evaluation tables and graphs.

## Models Used
### Phase 1
- Logistic Regression
- Random Forest
- Decision Tree

### Phase 2
- Poisson with equal weighted strengths
- Poisson with recency weighted strengths
- Poisson with form weighted strengths
- Poisson with opposition strength factored strengths

### Phase 3
- Laplace Distribution
- Negative Binomial
- Bayesian Poisson

## Evaluation
The evaluation takes data from 4 different seasons to see how models compare across different seasons and across all seasons combined.

### Metrics
- Log Loss
- Brier Score
- Accuracy

## Results
Every step in the data pipeline generates a CSV so we can validate the data flow. The Laplace model on average performed the best, with the Negative Binomial model also showing promising accuracy.
