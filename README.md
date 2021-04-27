# Taxi-Demand-Regression-with-Machine-Learning

## Introduction
This repository contains a Machine Learning model that predicts taxi demand in different times of the day.

## Dataset
The dataset used to train this model can be found in:

* [`./NYCdata2015/`](./NYCdata2015)

It records the Taxi demand and its corresponding temprature, wind speed, day of week and hour throughout the year of 2015, where demand was recorded 24 
times each day in New York City.


The dataset is broken down into 30 clusters where each cluster holds records of taxi demand in a different part of NYC.

__Note:__ the data is not yet cleaned and prepared for training a Machine Learning model except for `cluster04.csv` which I extracted out of the `./NYCdata2015` directory
and separated into data: `cluster04.csv` and targets: `cluster04targets.csv`

