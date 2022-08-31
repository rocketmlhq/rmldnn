
def main_optuna(command,n_epochs,n_trials,optimizers,losses,layers_file,lr_scheduler,transfer_learning,lr_range_start=0.001,lr_range_end=0.001,lr_scheduler_type='Exponential',lr_scheduler_gamma=0.95,tl_file=''):

    import os
    import json
    import shutil

    l = []
    prev_acc = [0]

    def create_optimizer(trial):
        optimizer_options = optimizers
        optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
        return optimizer_selected


    def create_learning_rate(trial):
        lr_selected = trial.suggest_float("learning_rate", lr_range_start, lr_range_end, log=True)
        return lr_selected


    def create_loss(trial):
        loss_options = losses
        loss_selected = trial.suggest_categorical("loss", loss_options)
        return loss_selected


    def objective(trial):
        optimizer = create_optimizer(trial)
        if lr_scheduler:
            lr = create_learning_rate(trial)
        else:
            lr = lr_range_start
        loss = create_loss(trial)
        config = {
            "neural_network": {
                "outfile": "out.txt",
                "num_epochs": n_epochs,
                "layers": "./{0}".format(layers_file),
                "checkpoints": {
                    "save": "temp_model/",
                    "interval": n_epochs
                },
                "data": {
                    "type": "images",
                    "input_path": "./data/train/inputs/",
                    "target_path": "./data/train/targets/",
                    "test_input_path": "./data/test/inputs/",
                    "test_target_path": "./data/test/targets/",
                    "batch_size": 32,
                    "test_batch_size": 64,
                    "preload": True,
                    "target_grayscale": True,
                    "target_is_mask": True,
                    "transforms": [
                        {"resize": [256, 256]}
                    ]
                },
                "optimizer": {
                    "type": optimizer,
                    "learning_rate": lr
                },
                "loss": {
                    "function": loss,
                    "source": "sigmoid"
                }
            }
        }

        if transfer_learning:
            config["neural_network"]["checkpoints"]["load"] = "./{0}".format(tl_file)

        if lr_scheduler:
            config["neural_network"]["optimizer"]["lr_scheduler"] = {"type": "{0}".format(lr_scheduler_type),"gamma": lr_scheduler_gamma,"verbose": True}

        with open("config.json", "w") as outfile:
            json.dump(config, outfile)

        os.system(command)
        with open('out_test.txt') as f:
            lines = f.readlines()

        acc = float(lines[-1].split('Accuracy: ')[1].replace('\n', ''))
        max_acc = acc

        for line in lines:
            max_acc = max(max_acc, float(line.split('Accuracy: ')[1].replace('\n', '')))

        d = {"max_acc": max_acc, "optimizer": optimizer, "lr": lr, "loss": loss, "accuracy_last_epoch": acc}
        print(d)
        l.append(d)
        os.remove('out_test.txt')
        os.remove('out_train.txt')
        os.remove('config.json')

        if(max_acc > prev_acc[-1]):
            shutil.rmtree("best_model", ignore_errors=True)
            os.mkdir("best_model")
            shutil.move("temp_model/model_checkpoint_{0}.pt".format(n_epochs),"best_model")
            prev_acc[-1] = max_acc

        shutil.rmtree("temp_model", ignore_errors=True)
        return max_acc

    import optuna

    os.mkdir("best_model")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open("list_acc.json", "w") as outfile:
        json.dump(l, outfile, indent=4, separators=(',', ': '))


