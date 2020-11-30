from easydict import EasyDict as edict

config = edict()


def generate_config(loss_name, dataset, network):
    # loss
    config.memonger = False
    config.embedding_size = 512
    config.bn_mom = 0.9
    config.workspace = 256
    config.net_se = 0
    config.net_act = 'prelu'
    config.net_unit = 3
    config.net_input = 1
    config.net_output = 'E'

    config.frequent = 200
    config.verbose_flag = False
    config.verbose = 2000
    config.data_shape = (3, 256, 128)

    if loss_name == 'arcface':
        config.loss_s = 64.0
        config.loss_m1 = 1.0
        config.loss_m2 = 0.5
        config.loss_m3 = 0.0
    elif loss_name == 'cosface':
        config.loss_s = 64.0
        config.loss_m1 = 1.0
        config.loss_m2 = 0.0
        config.loss_m3 = 0.40
    elif loss_name == 'curricular':
        config.loss_s = 64.0
        config.loss_m1 = 1.0
        config.loss_m2 = 0.5
        config.loss_m3 = 0.01

    if dataset == 'webface':

        config.lr_steps = '20000,28000'
        # config.lr_steps = '20000,28000,32000'
        config.sample_ratio = 1.0
        config.fp16 = False
        config.scheduler_type = 'sgd'
        #
        config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
        config.rec_list = [
            '/root/face_datasets//webface/train.rec'
        ]
        #
        config.head_name_list = ['webface']
        config.memory_lr_scale_list = [1.0]
        config.num_classes_list = [10575]
        config.batch_size = 64
        config.max_update = 32000
        # config.max_update = 36000
        config.warmup_steps = config.max_update // 5
        config.backbone_lr = 0.1
        config.backbone_final_lr = config.backbone_lr / 1000
        config.memory_bank_lr = 0.1
        config.memory_bank_final_lr = config.backbone_lr / 1000

    ## MS1MV2
    elif dataset == 'emore':
        config.lr_steps = '100000,160000'
        config.sample_ratio = 1.0
        config.fp16 = True
        config.scheduler_type = 'sgd'
        #
        config.val_targets = ['agedb_30', 'calfw', 'cfp_ff', 'cplfw', 'lfw', 'vgg2_fp']
        config.rec_list = [
            '/anxiang/datasets/faces_emore/train.rec'
        ]
        #
        config.head_name_list = ['emore']
        config.memory_lr_scale_list = [1.0]
        config.num_classes_list = [85742]
        config.batch_size = 64
        config.max_update = 180000
        config.warmup_steps = config.max_update // 5
        config.backbone_lr = 0.1
        config.backbone_final_lr = config.backbone_lr / 1000
        config.memory_bank_lr = 0.1
        config.memory_bank_final_lr = config.backbone_lr / 1000

        ## webface_emore
    elif dataset == 'webface_emore':
        config.lr_steps = '140000,200000'
        config.sample_ratio = 1.0
        config.fp16 = False
        config.scheduler_type = 'sgd'
        #
        config.val_targets = ['agedb_30', 'calfw', 'cfp_ff', 'cplfw', 'lfw', 'vgg2_fp']
        config.rec_list = [
            '/root/face_datasets/faces_emore/train.rec',
             '/root/face_datasets/webface/train.rec',
        ]
        #
        config.head_name_list = ['webface_emore']
        config.memory_lr_scale_list = [1.0]
        config.num_classes_list = [96317]
        config.batch_size = 64
        config.max_update = 220000
        config.warmup_steps = config.max_update // 5
        config.backbone_lr = 0.1
        config.backbone_final_lr = config.backbone_lr / 1000
        config.memory_bank_lr = 0.1
        config.memory_bank_final_lr = config.backbone_lr / 1000

     ## zunyi_set_260w
    elif dataset == 'zunyi_set_260w':
        config.lr_steps = '140000,200000'
        config.sample_ratio = 1.0
        config.fp16 = True
        config.scheduler_type = 'sgd'
        #
        config.val_targets = ['agedb_30', 'calfw', 'cfp_ff', 'cplfw', 'lfw', 'vgg2_fp']
        config.rec_list = [
            '/home/ubuntu/reid_zunyi_260w/shuf_train_zunyi_set_260w.rec',
        ]
        #
        config.head_name_list = ['zunyi_set_260w']
        config.memory_lr_scale_list = [1.0]
        config.num_classes_list = [2600635]
        config.batch_size = 128
        config.max_update = 220000
        config.warmup_steps = config.max_update // 5
        config.backbone_lr = 0.1
        config.backbone_final_lr = config.backbone_lr / 1000
        config.memory_bank_lr = 0.1
        config.memory_bank_final_lr = config.backbone_lr / 1000

    elif dataset == 'market_ducket_cuhk03_person28w':
        config.lr_steps = '140000,200000'
        config.sample_ratio = 1.0
        config.fp16 = True
        config.scheduler_type = 'sgd'
        #
        config.val_targets = ['agedb_30', 'calfw', 'cfp_ff', 'cplfw', 'lfw', 'vgg2_fp']
        config.rec_list = [
            '/home/ubuntu/market_ducket_cuhk03_person28w/market_ducket_cuhk03_person28w.rec',
        ]
        #
        config.head_name_list = ['market_ducket_cuhk03_person28w']
        config.memory_lr_scale_list = [1.0]
        config.num_classes_list = [112504]
        config.batch_size = 128
        config.max_update = 220000
        config.warmup_steps = config.max_update // 5
        config.backbone_lr = 0.1
        config.backbone_final_lr = config.backbone_lr / 1000
        config.memory_bank_lr = 0.1
        config.memory_bank_final_lr = config.backbone_lr / 1000

    # network
    if network == 'r100':
        config.net_name = 'fresnet'
        config.num_layers = 100
    elif network == 'r122':
        config.net_name = 'fresnet'
        config.num_layers = 122
    elif network == 'r50':
        config.net_name = 'fresnet'
        config.num_layers = 50
    elif network == 'rx101':
        config.net_name = 'fresnext'
        config.num_layers = 101
    elif network == 'rx50':
        config.net_name = 'fresnext'
        config.num_layers = 50
