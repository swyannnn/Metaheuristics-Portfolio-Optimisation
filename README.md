
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

4. Run the Main Script

```
cd path/to/AIM
python main.py
```

Replace `path/to/AIM` to your desired path.

5. View Tensorboard

Enter this command in terminal:
```
tensorboard --logdir "outputs" --port 8000
```
The tensorboard should start on port 6007. Open your browser and navigate to http://localhost:8000/ to access the application.

You can expect to see the best fitness plots of wach algorithms.

## What output you can expect to see:
In `./outputs` folder, you can see 

1) the comparison of EF and the overlay plots of respective algorithm's best portfolios. Namely `GA.png`, `SA.png`, `PSO.png`

2) Violin plot of the Tracking Error. Namely `annotated_violin_plot.png`

