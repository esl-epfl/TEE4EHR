import Main

import optuna
from optuna.integration import PyTorchLightningPruningCallback
# from optuna.integration.tensorboard import TensorBoardCallback



# from optuna.integration.wandb import WeightsAndBiasesCallback


# import os
# import wandb
# os.environ["WANDB_API_KEY"] = "0f780ac8a470afe6cb7fc474ff3794772c660465"
# # os.environ["WANDB_START_METHOD"] = "thread"

# wandb_kwargs = {"project": "TEDAM3"}
# wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

# tensorboard_callback = TensorBoardCallback("logs/optuna/", metric_name="accuracy")








# if os.path.exists(data['add_data']+'Optuna/') and os.path.isdir(data['add_data']+'Optuna/'):
#     shutil.rmtree(data['add_data']+'Optuna/')
    
# print(f"######################### TENSORBOARD #######################\ntensorboard --logdir={data['add_data'] + 'Optuna/'} --port 1374")

my_pruner = optuna.pruners.PercentilePruner(percentile=20,n_startup_trials=5,n_warmup_steps=15)
# my_pruner = optuna.pruners.ThresholdPruner(lower=0.55, upper=1, n_warmup_steps=10, interval_steps=1)


study = optuna.create_study(direction="maximize",
                            study_name=None,
                            sampler=optuna.samplers.TPESampler(),
                            pruner=my_pruner)
study.optimize(Main.main, timeout=12*3600,n_trials=50 )# ,callbacks=[wandbc]