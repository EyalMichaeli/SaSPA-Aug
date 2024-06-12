class BaseArgs:
    def __init__(self):
        # needs to be given:
        self.seed = 1
        self.gpu_id = 0
        self.logdir = None
        self.dataset = None
        self.train_sample_ratio = None
        self.aug_json = None 
        self.aug_sample_ratio = None

        # has default
        self.learning_rate = None
        self.batch_size = None
        self.weight_decay = None
        self.epochs = None
        self.limit_aug_per_image = None
        self.special_aug = None
        self.stop_aug_after_epoch = None
        self.use_target_soft_cross_entropy = False
        self.dont_use_wsdan = False
        self.use_cutmix = False
        self.use_target_soft_cross_entropy = False
