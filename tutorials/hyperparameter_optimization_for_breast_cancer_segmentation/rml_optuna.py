
def main_optuna(command,n_epochs,n_trials,optimizers,losses,layers_file,lr_scheduler,transfer_learning="",lr_range_start=0.001,lr_range_end=0.001,lr_scheduler_type='Exponential',lr_scheduler_gamma=0.95):

    import os
    import json
    import shutil
    import re

    list_dic = []

    acc_scores=[0]

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
                "layers": "{0}".format(layers_file),
                "checkpoints": {
                    "save": "temp_model/",
                    "interval": 1
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

        if transfer_learning!="":
            config["neural_network"]["checkpoints"]["load"] = "{0}".format(transfer_learning)

        if lr_scheduler:
            config["neural_network"]["optimizer"]["lr_scheduler"] = {"type": "{0}".format(lr_scheduler_type),"gamma": lr_scheduler_gamma,"verbose": True}

        with open("config.json", "w") as outfile:
            json.dump(config, outfile, indent=4, separators=(',', ': '))

        os.system(command)

        with open('out_test.txt') as f:
            lines = f.readlines()


        max_acc = 0
        epoch_max_acc = n_epochs

        for line in lines:
            numbers=re.findall(r"[0-9]*\.?[0-9]+", line)
            current_acc=float(numbers[-1])
            current_epoch=int(numbers[0])
            if max_acc < current_acc:
                max_acc = current_acc
                epoch_max_acc=current_epoch

        d = {"max_acc": max_acc, "optimizer": optimizer, "lr": lr, "loss": loss, "epoch_max_acc":epoch_max_acc}
        print(d)
        list_dic.append(d)
        os.remove('out_test.txt')
        os.remove('out_train.txt')
        os.remove('config.json')

        if(max_acc > max(acc_scores)):
            shutil.rmtree("best_model", ignore_errors=True)
            os.mkdir("best_model")
            shutil.move("temp_model/model_checkpoint_{0}.pt".format(epoch_max_acc),"best_model")
            os.rename("best_model/model_checkpoint_{0}.pt".format(epoch_max_acc), "best_model/best_model_file.pt")

        acc_scores.append(max_acc)

        shutil.rmtree("temp_model", ignore_errors=True)
        return max_acc

    import optuna

    try:
        os.mkdir("best_model")
    except Exception as e:
        shutil.rmtree("best_model", ignore_errors=True)
        os.mkdir("best_model")

    try:
        shutil.rmtree("temp_model", ignore_errors=True)
    except Exception as e:
        pass

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
        json.dump(list_dic, outfile, indent=4, separators=(',', ': '))

