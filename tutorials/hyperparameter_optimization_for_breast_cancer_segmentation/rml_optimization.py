import typer
from tabulate import tabulate
import rml_optuna

app = typer.Typer(help="CLI based on RMLDNN for automating hyper parameter optimization")


@app.command()
def rml_typer(
        num_epochs:int            = typer.Option(..., "--num-epochs", "-ne", help="Number of epochs per trial"),
        num_trials:int            = typer.Option(..., "--num-trials", "-nt", help="Number of trials"),
        docker_image:str          = typer.Option("rocketml/rmldnn:latest", "--docker-image", "-docker", help="Docker image path"),
        singularity_image:str     = typer.Option("None", "--singularity-image", "-singularity", help="Singularity image path"),
        gpu:bool                  = typer.Option(True, help="Whether to run on GPUs (instead of CPUs)"),
        num_procs:int             = typer.Option(1, "--num_procs", help="Number of processes (or GPUs) for parallel runs"),
        optimizers:str            = typer.Option(..., "--optimizers", "-o", help="Comma-separated list of optimizers (supported: SGD, RMSprop, Adagrad, Adam, AdamW, LARS, LAMB, Hessian)"),
        loss:str                  = typer.Option(..., "--loss", "-l", help="Comma-separated list of loss functions (supported: Dice, BCE, NLL, MSE, Jaccard, Focal, Lovasz)"),
        layers:str                = typer.Option("./layers.json", help="Network definition JSON file"),
        lr_range_min:float        = typer.Option(0.001, "--lr-range-min", "-lr-min", help="Lowest learning-rate value to try"),
        lr_range_max:float        = typer.Option(0.01, "--lr-range-max", "-lr-max", help="Highest learning-rate value to try"),
        lr_scheduler:bool         = typer.Option(False, help="Whether to use an exponential learning-rate scheduler"),
        lr_scheduler_gamma:float  = typer.Option(0.95, "--lr-scheduler-gamma", "-lr-gamma", help="LR change rate gamma (if using scheduler)"),
        transfer_learning:str     = typer.Option("", "--transfer-learning", "-tl", help="Model to load for transfer-learning")):

    optimizers = optimizers.split(',')
    loss = loss.split(',')

    lr_scheduler_type="Exponential" # Only type supported for now

    if singularity_image == "None":
        command = "docker run -u $(id -u):$(id -g) -v ${{PWD}}:/home/ubuntu -w /home/ubuntu --rm {0}".format(docker_image)
        index = command.find('-u')
        if gpu:
            command = command[:index] + "--gpus=all " + command[index:]
        if num_procs > 1:
            command = command + " mpirun -np {0}".format(num_procs)
            command = command[:index] + " --cap-add=SYS_PTRACE " + command[index:]
            if not gpu:
                command = command + " --bind-to none -x OMP_NUM_THREADS={0}".format(int(num_procs) * 2)
            else:
                cuda_devices = ','.join([str(i) for i in range(num_procs)])
                command = command + " -x CUDA_VISIBLE_DEVICES={0}".format(cuda_devices)

    else:
        command = "singularity exec {0}".format(singularity_image)
        if gpu:
            index=command.find(singularity_image)
            command = command[:index] + "--nv " + command[index:]

        if num_procs > 1:
            command += " mpirun -np {0}".format(num_procs)
            if not gpu:
                command = command + " --bind-to none -x OMP_NUM_THREADS={0} ".format(int(num_procs) * 2)
            else:
                cuda_devices = ','.join([str(i) for i in range(num_procs)])
                command = command + " -x CUDA_VISIBLE_DEVICES={1} ".format(num_procs, cuda_devices)

    command += " rmldnn --config=config.json"

    dic_values={}

    dic_values["command"] = command
    dic_values["num_epochs"] = num_epochs
    dic_values["num_trials"] = num_trials
    dic_values["layers"] = layers
    dic_values["optimizers"] = optimizers
    dic_values["loss"] = loss
    dic_values["lr_range_min"] = lr_range_min
    dic_values["lr_range_max"] = lr_range_max
    dic_values["lr_scheduler"] = lr_scheduler

    if lr_scheduler:
        dic_values["lr_scheduler_type"] = lr_scheduler_type
        dic_values["lr_scheduler_gamma"] = lr_scheduler_gamma

    if transfer_learning != "":
        dic_values["transfer_learning"] = transfer_learning

    headers = ["Parameter", "value"]
    print(tabulate([(k,v) for k, v in dic_values.items()], headers=headers, tablefmt='grid'))
    rml_optuna.main_optuna(command, num_epochs, num_trials, optimizers, loss, layers, lr_scheduler, transfer_learning, lr_range_min, lr_range_max, lr_scheduler_type, lr_scheduler_gamma)

if __name__ == "__main__":

    app()
