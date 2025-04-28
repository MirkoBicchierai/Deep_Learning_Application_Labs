
"""
Function used to define the type of an attribute inside of the parser for the main, to parse for example '--Residual true'
"""
import wandb


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

    exp_name = "_New"

    wandb.init(
        project="DLA_LAB_3",
        config=vars(args),
        name=exp_name,
    )

