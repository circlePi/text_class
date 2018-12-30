"""学习率衰减策略"""

def exponential_decay(optimizer, epoch):
    pass


def custom_decay(optimizer, epoch):
    if epoch % 2 != 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group['lr'] * 0.1
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = 0.001


def cosine_anneal_decay(optimizer, epoch):
    pass


def lr_update(optimizer, epoch, lr_decay_mode):
    if lr_decay_mode == "constant":
        pass
    elif lr_decay_mode == "exponential_decay":
        exponential_decay(optimizer, epoch)
    elif lr_decay_mode == "cosine_anneal_decay":
        cosine_anneal_decay(optimizer, epoch)
    elif lr_decay_mode == "custom_decay":
        custom_decay(optimizer, epoch)
    else:
        raise TypeError("Unknown lr update mode")
