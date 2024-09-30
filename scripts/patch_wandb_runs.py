import wandb


api = wandb.Api()
runs = api.runs("rylan/ft_collapse")

run_ids = {
    "cixxjhe4",
    "4cb33jbf",
    "wwu6cqq0",
    "ybmfn7yg",
    "qjq6joon",
    "q0fiw8z4",
    "m22ugw89",
    "am1z6job",
    "etoybxj3",
    "9n9jebt0",
    "1ushvjpd",
    "sh63rml3",
    "qffsvnw4",
    "9026kvk8",
    "37o5gbih",
    "ymi9r9o0",
    "wszri0vs",
    "70cvdrxk",
}

runs = [run for run in runs if run.id in run_ids]

for run in runs:
    # Update run config to contain rloo_trainer_config.gold_reward_model_eval_steps = 100
    run.config["paradigm"] = "Accumulate-Subsample"
    run.update()

print("Patched runs' configs.")
