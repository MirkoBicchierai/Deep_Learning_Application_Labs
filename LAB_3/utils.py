import wandb

"""
Function used to define the type of an attribute inside of the parser for the main, to parse for example '--lora true'
"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False

    return None


"""
Function to define the wandb parameters for the main.py 
"""
def config_loggers(args):

    if args.lora:
        prex = "LORA" + "-A:" + str(args.lora_alpha) + "-R:" + str(args.lora_rank)
    else:
        prex = "FULL-MODEL"

    formatted_lr = f"{args.lr:.0e}".replace("e+00", "e+0").replace("e-00", "e-0")
    exp_name = prex + "-lr:" + formatted_lr + "-batch_size:" + str(args.batch_size)

    wandb.init(
        project="DLA_LAB_3",
        config=vars(args),
        name=exp_name,
    )

