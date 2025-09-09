import torch
from torch import optim
import lightning as L
from LightningModule import SpeechReModel
from DataModule import SpeechReDatamodule
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger,WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler

class Hypernum:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def get_attributes_as_string(self):
        # 获取属性字典
        attributes = self.__dict__
        
        # 生成最终的字符串列表
        result = []
        for key, value in attributes.items():
            # 忽略特定的属性
            if key in ['data_path', 'audio_file', 'whisper_model_dir',"data_type"]:
                continue

            # 如果值是列表，则将其转为逗号分隔的字符串
            if isinstance(value, list):
                value = ",".join(map(str, value))  # 转为字符串并用逗号分隔
            else:
                value = str(value)  # 其他类型直接转换为字符串
            
            # 将键和值用"_"连接
            result.append(f"{key}_{value}")
        
        # 将所有键值对连接成一个大字符串
        return "_".join(result)
    def get_all_attributes(self):
        return self.__dict__  # 或者使用 vars(self)

hypernum = Hypernum(
        data_path = r"Datasets/speech_conll04",
        audio_file = r"/root/autodl-tmp/speech_conll04/audio",
        whisper_model_dir = r"PretrainedSpeechModel/whisper-small.en",
        data_type = "train",
        train_order_views = [0,1,2,3,4,5],
        predict_order_views = [0,1,2,3,4,5],
        del_no_relation_type = False,
        add_entity_target = False,
        train_data_mix = 'random',#random or concat
        batch_size = 8,
        epoch_num = 10,
        warmup_rate = 0.02,
        val_check_interval = 0.2,
        learning_rate =5e-5,
        rc_lr_factor = 1,
        seed = 42,
        num_beams = 3,
        decoder_max_length = 448,
        precision = '16-mixed',
        gpu_device = [0],
        dataloader_numworkers = 8,
        vote_threshold = 2
    )
seed_everything(hypernum.seed, workers=True)

data_module = SpeechReDatamodule(hypernum)#data_module必须在model前初始化，设计到tokenizer的传递与改变
model = SpeechReModel.load_from_checkpoint(r"/root/autodl-tmp/checkpoints/conll04-epoch=25-F1-vote投票后-三元组F1=0.29.ckpt",hypernum = hypernum)

#logger
tblogger = TensorBoardLogger("/root/tf-logs", name="dev_test_model",default_hp_metric=False)

profiler = PyTorchProfiler()

lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(dirpath="/root/autodl-tmp/checkpoints", 
                                      filename='imner-{epoch:02d}-{total_loss_epoch:.2f}',
                                      save_top_k=1, 
                                      monitor='hrt-F1',
                                      mode = 'max',
                                      save_weights_only = True
                                      )



trainer = L.Trainer(
                    max_epochs=hypernum.epoch_num,
                    logger = tblogger,
                    callbacks=[lr_monitor,checkpoint_callback],
                    devices=hypernum.gpu_device,
                    enable_checkpointing=True,
                    precision = hypernum.precision,
                    check_val_every_n_epoch=1,
                    #profiler=profiler,
                    #limit_val_batches = 5

)

trainer.test(model=model, datamodule=data_module)