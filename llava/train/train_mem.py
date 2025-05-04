from llava.train.train import train

from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

if __name__ == "__main__":
    import wandb
    wandb.init(mode="disabled")
    train()
