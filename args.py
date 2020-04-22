class Args():
    def __init__(self):
        self.cuda = False
        self.save_best = True
        self.batch_size = 256
        self.lr = 0.01
        self.epochs = 200
        self.log_interval = 1
        self.dev_interval = 1
        self.save_interval = 20
        self.dropout = 0.4
        self.save_dir = "save/"
        self.embed_num = 18240
        self.embed_dim = 50
        self.class_num = 5
        self.kernel_num = 256
        self.kernel_sizes = [2,3,4]
        self.early_stop = 200


