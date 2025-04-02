
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
tensorboard --logdir "outputs"
```
Then visit the localhost stated in terminal.
