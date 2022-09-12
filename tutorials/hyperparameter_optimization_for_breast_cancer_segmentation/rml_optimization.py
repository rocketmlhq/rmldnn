import typer
from tabulate import tabulate
import rml_optuna

app = typer.Typer(help="CLI based on RMLDNN for automating hyper parameter optimization")


@app.command()
def rml_typer(
        num_epochs:int            = typer.Option(..., "--num-epochs", "-ne"),
        num_trials:int            = typer.Option(..., "--num-trials", "-nt"),
        docker_image:str          = typer.Option("rocketml/rmldnn:latest", "--docker-image", "-docker", help="Enter docker image path"),
        singularity_image:str     = typer.Option("None", "--singularity-image", "-singularity", help="Enter singularity image path"),
        gpu:bool                  = False,
        multi_core:int           = typer.Option(1, "--multi-core", help="Enter number of process"),
        optimizers:str            = typer.Option(..., "--optimizers", "-o", help="Enter optimizers comma seperated (allowed values : adam, SGD, RMSprop, Adagrad, Adam and AdamW, LARS, LAMB, Hessian)"),
        loss:str                  = typer.Option(..., "--loss", "-l", help="Enter loss functions comma seperated(allowed values: dice, bce, nll, mse, Jaccard, Focal, Lovasz, Wasserstein, YOLOv3, Burgers_pde, Poisson2D_pde, Poisson3D_pde)"),
        lr:float                  = typer.Option(0.001, "--learning-rate", "-lr", help="Enter Learning Rate"),
        layers:str                = "layers.json",
        lr_scheduler:bool         = False,
        lr_range_start:float      = typer.Option(0.001, "--lr-range-start", "-lr-start", help="Enter Learning Rate scheduler start value"),
        lr_range_end:float        = typer.Option(0.001, "--lr-range-end", "-lr-end", help="Enter Learning Rate scheduler end value"),
        lr_scheduler_gamma:float  = typer.Option(0.95, "--lr-scheduler-gamma", "-lr-gamma", help="Enter Learning Rate scheduler gamma value"),
        transfer_learning:str    = typer.Option("", "--transfer-learning", "-tl", help="Enter file name using which you want to implement transfer learning")):

    optimizers = optimizers.split(',')
    loss = loss.split(',')

    lr_scheduler_type="Exponential"

    if singularity_image == "None":
        command = "docker run -u $(id -u):$(id -g) -v ${{PWD}}:/home/ubuntu -w /home/ubuntu --rm {0}".format(docker_image)
        index = command.find('-u')
        if gpu:
            command = command[:index] + "--gpus=all " + command[index:]
        if multi_core > 1:
            command = command + " mpirun -np {0}".format(multi_core)
            command = command[:index] + " --cap-add=SYS_PTRACE " + command[index:]
            if not gpu:
                command = command + " --bind-to none -x OMP_NUM_THREADS={0}".format(int(multi_core) * 2)
            else:
                cuda_devices = ','.join([str(i) for i in range(multi_core)])
                command = command + " -x CUDA_VISIBLE_DEVICES={0}".format(cuda_devices)

    else:
        command = "singularity exec {0}".format(singularity_image)
        if gpu:
            index=command.find(singularity_image)
            command = command[:index] + "--nv " + command[index:]

        if multi_core > 1:
            command += " mpirun -np {0}".format(multi_core)
            if not gpu:
                command = command + " --bind-to none -x OMP_NUM_THREADS={0} ".format(int(multi_core) * 2)
            else:
                cuda_devices = ','.join([str(i) for i in range(multi_core)])
                command = command + " -x CUDA_VISIBLE_DEVICES={1} ".format(multi_core, cuda_devices)

    command += " rmldnn --config=config.json"

    dic_values={}

    dic_values["command"] = command
    dic_values["num_epochs"] = num_epochs
    dic_values["num_trials"] = num_trials
    dic_values["layers"] = layers
    dic_values["optimizers"] = optimizers
    dic_values["loss"] = loss


    if lr_range_start == lr_range_end:
        lr_scheduler = False

    if lr_scheduler:

        dic_values["lr_range_start"] = lr_range_start
        dic_values["lr_range_end"] = lr_range_end
        dic_values["lr_scheduler_type"] = lr_scheduler_type
        dic_values["lr_scheduler_gamma"] = lr_scheduler_gamma

    else:
        dic_values["learning_rate"] = lr


    if transfer_learning != "":
        dic_values["transfer_learning"] = transfer_learning

    headers = ["Parameter", "value"]
    print(tabulate([(k,v) for k, v in dic_values.items()], headers=headers, tablefmt='grid'))
    rml_optuna.main_optuna(command,num_epochs,num_trials,optimizers,loss,layers,lr_scheduler,transfer_learning,lr_range_start,lr_range_end,lr_scheduler_type,lr_scheduler_gamma)

if __name__ == "__main__":

    app()
