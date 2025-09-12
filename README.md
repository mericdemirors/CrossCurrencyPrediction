# Cross-Coin Cryptocurrency Price Prediction

This project explores cryptocurrency price prediction using deep learning models and evaluates them against benchmark and novel trading strategies. The workflow involves data collection, preprocessing, model training, and strategy evaluation.

Check the summarizing presentation at [project_presentation.pdf](project_presentation.pdf)

## Repository Structure

### Jupyter Notebooks
These notebooks follow the project pipeline from raw data collection to model experimentation:

* [01_APIs.ipynb](01_APIs.ipynb) – Try out new APIs to collect data
* [02_collect_data.ipynb](02_collect_data.ipynb) – Collect the data and save it
* [03_plot.ipynb](03_plot.ipynb) – Check the properties of the data
* [04_data_cleaning.ipynb](04_data_cleaning.ipynb) – Process and save the cleaned dataset
* [05_dataset.ipynb](05_dataset.ipynb) – Try out dataset formulations
* [06_models.ipynb](06_models.ipynb) – Experiment with model architectures
* [10_strategies.ipynb](10_strategies.ipynb) – Test trading strategies

### Training & Evaluation Scripts (train_scripts/)
This directory contains scripts and modular components for training and evaluation:

* [07_train.ipynb](train_scripts/07_train.ipynb) – Main notebook to run all experiments
* [08_evaluation.ipynb](train_scripts/08_evaluation.ipynb) – Experiment with evaluation techniques
* [09_profit_inference.ipynb](train_scripts/09_profit_inference.ipynb) – Explore profit inference techniques
* [dataset_classes/](train_scripts/dataset_classes) – Contains dataset definitions
* [import_dataset.py](train_scripts/import_dataset.py) – Imports selected dataset class
* [additional_losses/](train_scripts/additional_losses) – Contains additional/custom loss functions
* [import_loss.py](train_scripts/import_loss.py) – Imports selected loss function
* [model_classes/](train_scripts/model_classes) – Contains model architectures
* [import_model.py](train_scripts/import_model.py) – Imports selected model
* [train.py](train_scripts/train.py) – Main script for model training
* [evaluate.py](train_scripts/evaluate.py) – Script to generate evaluation plots
* [profit_inference.py](train_scripts/profit_inference.py) – Evaluates models with the Toast Bread strategy
* [trading_agents.py](train_scripts/trading_agents.py) – Collection of trading strategies

## Project Flow
* Data Collection – APIs are tested and used to gather OHLC data.
* Preprocessing – Log returns and quantile normalization are applied to handle domain shifts across coins.
* Datasets – Multiple dataset classes explore different ways of encoding OHLC and low-high information.
* Models – Sequence-to-sequence architectures (GRU, LSTM, Transformer), TCN and cross-attention variants are tested.
* Training & Evaluation – Dataset-model pairs are trained further and evaluated with loss metrics and trading simulations.
* Trading Strategies – Benchmarks plus a novel Toast Bread strategy, which plans buy/sell orders around predicted future prices.

## Results & Insights
* Encoding low–high relationships in normalization improved results.
* LSTM and GRU models showed the best trainability.
* The Toast Bread strategy consistently outperformed traditional strategies when paired with prediction models.
* Profitable cases include LSTM on BTC and BNB, achieving up to ~1.9x simulated returns, and almost x4 more profit than the best classical trading strategy.

## Future Work
* Extend to more coins and features.
* Explore invertible transforms (Fourier, Wavelet).
* Design datasets/models that better capture high–low order.
* Develop hybrid strategies using both past data and future predictions.

# Installation and Usage

Install the [requirements.txt](requirements.txt) and run the notebook [07_train.ipynb](train_scripts/07_train.ipynb) with the instructions.