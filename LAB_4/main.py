import wandb

def main():
    wandb.init(
        project="DLA_LAB_4",
        name="experiment_stillness_reward",
        config={
            "learning_rate": 1e-3,
            "epochs": 4,
        }
    )

if __name__ == "__main__":
    main()