exp_name = "sample_train"
filter_class = 0  # [relax, open, close]
eval_interval = 1000  # checkpoint interval, how often the checkpoint is saved
eval_iters = 200
log_interval = 10  # don't print too often

always_save_checkpoint = True

wandb_log = True  # override via command line if you like
wandb_project = "chatemg"

gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters
split = 0.8  # 80% train, 20% val

# preprocessing
median_filter_size = 9

token_embedding_type = "basic_sum"
n_layer = 12
n_head = 8
n_embd = 256
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 30000
lr_decay_iters = 30000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially