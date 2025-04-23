
# Artificial Intelligience Method Coursework
This coursework uses a Conda environment to run the scripts. Follow the steps below to set up the environment, install dependencies, and run the main script.

## Setup Instructions

1. Create a Conda Environment

```
conda create -n AIM_cw_env python=3.13
```

2. Activate the Environment

```
conda activate AIM_cw_env
```

3. Install Dependencies

```
pip install -r requirements.txt
```

4. Enter the desired directory in terminal

```
cd path/to/AIM
```

5. Customise the desired configuration

`config/config.yaml` is the only place where we set all the parameter values.

6. Run the Main Script

```
python main.py
```

7. View Tensorboard

After running main script, insert this command in terminal:
```
tensorboard --logdir "outputs" --port 6006
```
The tensorboard should start on port 6006. Open your browser and navigate to http://localhost:6006/ to access the application.

You can expect to see the best fitness plots of each algorithms.

## What output you can expect to see:
In `./outputs` folder, you can see 

1) the comparison of efficient frontier and the overlay plots of respective algorithm's best portfolios. Namely `GA.png`, `SA.png`, `PSO.png`

2) Violin plot of the Tracking Error. Namely `annotated_violin_plot.png`

3) The logging of each algorithm's final results, including weight allocation, sharpe ratio, expected returm, volatility, etc.

