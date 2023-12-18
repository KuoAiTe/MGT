from pytorch_lightning import callbacks
def get_ckpt_save_path(model_name, dataset_info, current_fold):
    return f'{dataset_info.dataset_name}_{model_name}_{current_fold}_ntpp{dataset_info.num_tweets_per_period}_mnf{dataset_info.max_num_friends}_pim_{dataset_info.periods_in_months}'
def custom_callbacks(model_name, dataset_info, current_fold):
    return [
        callbacks.early_stopping.EarlyStopping(monitor = "val_loss", min_delta = 0.00, patience = 20, verbose = False, mode = "min"),
        callbacks.ModelCheckpoint(
            monitor = 'val_loss',
            mode = "min",
            dirpath = './checkpoints/',
            filename = get_ckpt_save_path(model_name, dataset_info, current_fold),#-{epoch:02d}-{val_loss:.2f}'
        )
    ]
