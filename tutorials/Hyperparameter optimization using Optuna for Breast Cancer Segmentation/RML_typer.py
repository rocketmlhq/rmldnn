import typer
from tabulate import tabulate
import RML_Optuna

app = typer.Typer(help="CLI based on RMLDNN for automating hyper parameter optimization")


@app.command()
def rml_typer(
        num_epochs:int          = typer.Option(..., "--num-epochs", "-ne"),
        num_trials:int          = typer.Option(..., "--num-trials", "-nt"),
        docker_image:str        = typer.Option("rocketml/rmldnn:latest", "--docker-image", "-docker", help="Enter docker image path"),
        singularity_image:str   = typer.Option("None", "--singularity-image", "-singularity", help="Enter singularity image path"),
        gpu:bool                = False,
        multi_core:bool         = False,
        optimizers:str          = typer.Option(..., "--optimizers", "-o", help="Enter optimizers comma seperated (allowed values : adam, SGD, RMSprop, Adagrad, Adam and AdamW, LARS, LAMB, Hessian)"),
        loss:str                = typer.Option(..., "--loss", "-l", help="Enter loss functions comma seperated(allowed values: dice, bce, nll, mse, Jaccard, Focal, Lovasz, Wasserstein, YOLOv3, Burgers_pde, Poisson2D_pde, Poisson3D_pde)"),
        lr:float                = typer.Option(0.001, "--learning-rate", "-lr", help="Enter Learning Rate"),
        layers:str              = "layers.json",
        lr_scheduler:bool       = False,
        transfer_learning:bool  = False):

    optimizers = optimizers.split(',')
    loss = loss.split(',')

    if multi_core:
        n_procs = int(typer.prompt("Enter number of processes"))

    if singularity_image == "None":
        command = "docker run -u $(id -u):$(id -g) -v ${{PWD}}:/home/ubuntu -w /home/ubuntu --rm {0}".format(docker_image)
        index = command.find('-u')
        if not multi_core:
            if gpu:
                command = command[:index] + " --gpus=all " + command[index:]
        else:
            command = command + " mpirun -np {0}".format(n_procs)
            command = command[:index] + " --cap-add=SYS_PTRACE " + command[index:]
            if not gpu:
                command = command + " --bind-to none -x OMP_NUM_THREADS={0}".format(int(n_procs) * 2)
            else:
                cuda_devices = ','.join([str(i) for i in range(n_procs)])
                command = command[:index] + " --gpus=all " + command[index:] + " -x CUDA_VISIBLE_DEVICES={0}".format(cuda_devices)

    else:
        command = "singularity exec {0}".format(singularity_image)
        if gpu:
            index=command.find(singularity_image)
            command = command[:index] + "--nv " + command[index:]
        if multi_core:
            command += " mpirun -np {0}".format(n_procs)
        if multi_core:
            if not gpu:
                command = command + " --bind-to none -x OMP_NUM_THREADS={0} ".format(int(n_procs) * 2)
            else:
                cuda_devices = ','.join([str(i) for i in range(n_procs)])
                command = command + " -x CUDA_VISIBLE_DEVICES={1} ".format(n_procs, cuda_devices)

    command += " rmldnn --config=config.json"

    if lr_scheduler:
        lr_range_start = float(typer.prompt("Enter LR start value"))
        lr_range_end = float(typer.prompt("Enter LR end value"))
        lr_scheduler_type = "Exponential"
        lr_scheduler_gamma = float(typer.prompt("Enter LR scheduler gamma value"))

    if transfer_learning:
        tl_file = typer.prompt("Enter model file to load")

    dic_values={}

    dic_values["command"] = command
    dic_values["num_epochs"] = num_epochs
    dic_values["num_trials"] = num_trials
    dic_values["layers"] = layers
    dic_values["optimizers"] = optimizers
    dic_values["loss"] = loss

    if lr_scheduler:

        dic_values["lr_range_start"] = lr_range_start
        dic_values["lr_range_end"] = lr_range_end
        dic_values["lr_scheduler_type"] = lr_scheduler_type
        dic_values["lr_scheduler_gamma"] = lr_scheduler_gamma

    else:
        dic_values["learning_rate"] = lr
        lr_range_start = lr_range_end = lr
        lr_scheduler_type = "Exponential"
        lr_scheduler_gamma = 0.95


    if transfer_learning:
        dic_values["tl_file"] = tl_file
    else:
        tl_file=""

    headers = ["Parameter", "value"]
    print(tabulate([(k,v) for k, v in dic_values.items()], headers=headers, tablefmt='grid'))
    RML_Optuna.main_optuna(command,num_epochs,num_trials,optimizers,loss,layers,lr_scheduler,transfer_learning,lr_range_start,lr_range_end,lr_scheduler_type,lr_scheduler_gamma,tl_file)

if __name__ == "__main__":

    app()
