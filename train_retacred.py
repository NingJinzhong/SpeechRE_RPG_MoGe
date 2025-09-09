import torch
from torch import optim
import lightning as L
from LightningModule import SpeechReModel
from DataModule import SpeechReDatamodule
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger,WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler
from swanlab.integration.pytorch_lightning import SwanLabLogger

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
#计划：增加是否有decoder提示的方式，checkpoint中的命名
hypernum = Hypernum(
        data_path = r"Datasets/speech_ReTracred",
        audio_file = r"/root/autodl-tmp/speech_ReTracred_audio/speech_ReTracred_audio",
        whisper_model_dir = r"PretrainedSpeechModel/whisper-medium.en",
        data_type = "train",
        train_order_views = [0,1,2,3,4,5],
        predict_order_views = [0,1,2,3,4,5],
        del_no_relation_type = False,
        add_entity_target = True,
        with_potional_rel_prompt = True,
        train_data_mix = 'random',#random or concat
        batch_size = 4,
        epoch_num = 15,
        warmup_rate = 0.05,
        val_check_interval = 0.2,
        learning_rate =1e-5,
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

unseen_elements = set(hypernum.predict_order_views) - set(hypernum.train_order_views)
if unseen_elements:
    raise ValueError(f"推理顺序视角中包含训练顺序视角中没有的视角: {unseen_elements}")
else:
    print("推理顺序视角都在训练顺序视角中出现过。")

data_module = SpeechReDatamodule(hypernum)#data_module必须在model前初始化，设计到tokenizer的传递与改变
model = SpeechReModel(hypernum)



if "conll04" in hypernum.data_path.lower():
    datasetname = "conll04"
if "retracred" in hypernum.data_path.lower():
    datasetname = "retraced"
if "cv17" in hypernum.data_path.lower():
    datasetname = "retraced"

if "tiny" in hypernum.whisper_model_dir.lower():
    asrmodel = "whisper-tiny"
if "base" in hypernum.whisper_model_dir.lower():
    asrmodel = "whisper-base"
if "small" in hypernum.whisper_model_dir.lower():
    asrmodel = "whisper-small"
if "medium" in hypernum.whisper_model_dir.lower():
    asrmodel = "whisper-medium"
if "large" in hypernum.whisper_model_dir.lower():
    asrmodel = "whisper-large" 

checkpoint_callback = ModelCheckpoint(dirpath="/root/autodl-tmp/checkpoints/medium", 
                                      filename='retraced-{epoch:02d}-F1-vote{投票后-三元组F1:.2f}',
                                      save_top_k=1, 
                                      monitor='投票后-三元组F1',
                                      mode = 'max',
                                      save_weights_only = True
                                      )

#logger
tblogger = TensorBoardLogger("/root/tf-logs", name="dev_test_model",default_hp_metric=False)
swanlab_logger = SwanLabLogger(
    project="ReTacred",
    experiment_name=hypernum.get_attributes_as_string(),
)
profiler = PyTorchProfiler()

lr_monitor = LearningRateMonitor(logging_interval='step')


trainer = L.Trainer(
                    max_epochs=hypernum.epoch_num,
                    logger = [tblogger,swanlab_logger],
                    callbacks=[lr_monitor,checkpoint_callback],
                    devices=hypernum.gpu_device,
                    enable_checkpointing=True,
                    precision = hypernum.precision,
                    accumulate_grad_batches=15,
                    check_val_every_n_epoch=1,
                    #limit_train_batches=0.1,
                    limit_val_batches=0.25
                    #val_check_interval = hypernum.val_check_interval
                    #profiler=profiler,
                    #limit_val_batches = 5

)
trainer.fit(model=model, datamodule=data_module)
#trainer.validate(model=model, datamodule=data_module)