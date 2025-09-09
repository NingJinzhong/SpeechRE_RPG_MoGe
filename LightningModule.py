import torch
import torch.nn as nn
import lightning as L
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup,\
                        get_linear_schedule_with_warmup,\
                        WhisperForConditionalGeneration,\
                        WhisperTokenizer
from DataModule import SpeechReDatamodule
from transformers.models.whisper.modeling_whisper import shift_tokens_right
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from LightningMetric import TripletF1,RelationF1,EntityF1,EncoderRelationF1
import time

class WhisperCNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(WhisperCNNClassifier, self).__init__()
        # 第 1 层卷积：输入通道 1，输出通道 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 尺寸减半
        )
        # 第 2 层卷积：输入通道 32，输出通道 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 尺寸再减半
        )
        # 第 3 层卷积：输入通道 64，输出通道 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第 4 层卷积：输入通道 128，输出通道 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 自适应全局平均池化，将特征映射变为 [B, 256, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层输出分类结果（假设有 num_classes 个类别）
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        参数 x 的形状为: [batch_size, seq_len, hidden_dim]
        """
        # 增加一个维度，变为 [batch_size, 1, seq_len, hidden_dim]
        # 将 seq_len 视为“时间”，hidden_dim 视为“特征”
        x = x.unsqueeze(1)  # [B, 1, S, D]

        x = self.conv1(x)   # [B, 32,  S/2,   D/2  ]
        x = self.conv2(x)   # [B, 64,  S/4,   D/4  ]
        x = self.conv3(x)   # [B, 128, S/8,   D/8  ]
        x = self.conv4(x)   # [B, 256, S/16,  D/16 ]

        # 全局平均池化到 [B, 256, 1, 1]
        x = self.global_pool(x)
        # 去除最后的空间维度 [B, 256]
        x = x.view(x.size(0), -1)
        # 全连接得到 logits
        logits = self.fc(x)
        return logits


class SpeechReModel(L.LightningModule):
    def __init__(self, hypernum):
        super().__init__()
        self.hypernum = hypernum
        self.WhisperGenModel = WhisperForConditionalGeneration.from_pretrained(hypernum.whisper_model_dir)
        self.WhisperEncoder = self.WhisperGenModel.get_encoder()
        self.WhisperDecoder = self.WhisperGenModel.get_decoder()
        self.WhisperDecoderLmhead = self.WhisperGenModel.get_output_embeddings()
        self.WhisperGenModel.resize_token_embeddings(len(self.hypernum.processor.tokenizer))
        self.WhisperGenModel.train()
        self.WhisperEncoder.train()
        self.WhisperDecoder.train()
        self.WhisperDecoderLmhead.train()
        self.loss_fct = CrossEntropyLoss()
        
        self.RelationClassifier = WhisperCNNClassifier(len(self.hypernum.relation_type_to_id)-1)
        self.relationclass_loss = nn.BCEWithLogitsLoss()
        self.all_order_views = ['hrt','htr','rht','rth','trh','thr']
        
        self.TripletF1_hrt = TripletF1(self.hypernum, "hrt")
        self.TripletF1_htr = TripletF1(self.hypernum, "htr")
        self.TripletF1_rht = TripletF1(self.hypernum, "rht")
        self.TripletF1_rth = TripletF1(self.hypernum, "rth")
        self.TripletF1_trh = TripletF1(self.hypernum, "trh")
        self.TripletF1_thr = TripletF1(self.hypernum, "thr")
        self.TripletF1_vote = TripletF1(self.hypernum, "vote")
        self.TripletF1MerticDic = dict(
            hrt = self.TripletF1_hrt,
            htr = self.TripletF1_htr,
            rht = self.TripletF1_rht,
            rth = self.TripletF1_rth,
            trh = self.TripletF1_trh,
            thr = self.TripletF1_thr
        )
        self.RelationF1_hrt = RelationF1(self.hypernum, "hrt")
        self.RelationF1_htr = RelationF1(self.hypernum, "htr")
        self.RelationF1_rht = RelationF1(self.hypernum, "rht")
        self.RelationF1_rth = RelationF1(self.hypernum, "rth")
        self.RelationF1_trh = RelationF1(self.hypernum, "trh")
        self.RelationF1_thr = RelationF1(self.hypernum, "thr")
        self.RelationF1_vote = RelationF1(self.hypernum, "vote")
        self.RelationF1MerticDic = dict(
            hrt = self.RelationF1_hrt,
            htr = self.RelationF1_htr,
            rht = self.RelationF1_rht,
            rth = self.RelationF1_rth,
            trh = self.RelationF1_trh,
            thr = self.RelationF1_thr
        )

        self.EntityF1_hrt = EntityF1(self.hypernum, "hrt")
        self.EntityF1_htr = EntityF1(self.hypernum, "htr")
        self.EntityF1_rht = EntityF1(self.hypernum, "rht")
        self.EntityF1_rth = EntityF1(self.hypernum, "rth")
        self.EntityF1_trh = EntityF1(self.hypernum, "trh")
        self.EntityF1_thr = EntityF1(self.hypernum, "thr")
        self.EntityF1_vote = EntityF1(self.hypernum, "vote")
        self.EntityF1MerticDic = dict(
            hrt = self.EntityF1_hrt,
            htr = self.EntityF1_htr,
            rht = self.EntityF1_rht,
            rth = self.EntityF1_rth,
            trh = self.EntityF1_trh,
            thr = self.EntityF1_thr
        )
        
        self.EncoderRelationF1_hrt = EncoderRelationF1(self.hypernum, "hrt")
        self.EncoderRelationF1_htr = EncoderRelationF1(self.hypernum, "htr")
        self.EncoderRelationF1_rht = EncoderRelationF1(self.hypernum, "rht")
        self.EncoderRelationF1_rth = EncoderRelationF1(self.hypernum, "rth")
        self.EncoderRelationF1_trh = EncoderRelationF1(self.hypernum, "trh")
        self.EncoderRelationF1_thr = EncoderRelationF1(self.hypernum, "thr")
        self.EncoderRelationMerticDic = dict(
            hrt = self.EncoderRelationF1_hrt,
            htr = self.EncoderRelationF1_htr,
            rht = self.EncoderRelationF1_rht,
            rth = self.EncoderRelationF1_rth,
            trh = self.EncoderRelationF1_trh,
            thr = self.EncoderRelationF1_thr
        )





        self.tokenizer = self.hypernum.processor.tokenizer
        self.order_map = dict(h = '<head_entity>',r = '<relation_type>',t = '<tail_entity>')
        self.order_to_id = {'<head_entity>':0,'<relation_type>':1,'<tail_entity>':2}
        self.id_to_order = {0:'<head_entity>',1:'<relation_type>',2:'<tail_entity>'}
        self.order_to_id_contain_e = dict(h = [0,1],r=[2],t = [3,4])

        self.save_hyperparameters(self.hypernum.get_all_attributes())

    def configure_optimizers(self):
        relation_module = self.RelationClassifier
        relation_param_ids = {id(p) for p in relation_module.parameters()}
        other_params = [p for p in self.parameters() if id(p) not in relation_param_ids]
        optimizer = AdamW([
        {"params": relation_module.parameters(), "lr": self.hypernum.learning_rate * self.hypernum.rc_lr_factor},
        {"params": other_params, "lr": self.hypernum.learning_rate}
        ])
        stepping_batches = self.trainer.estimated_stepping_batches
        num_warmup_steps = int((self.hypernum.warmup_rate)*stepping_batches)
        scheduler = get_constant_schedule_with_warmup(optimizer = optimizer,num_warmup_steps = num_warmup_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
            "scheduler":scheduler,
            "interval": "step",
            "frequency": 1,
            }
        }  
    def training_step(self, batch,batch_idx):
        stage = batch[5]
        whisper_encoder_inputs,whisper_encoder_relation_labels,whisper_decoder_inputs_train,whisper_decoder_inputs_predict = self.get_whisper_inputdata(batch)
        #encoder
        encoder_outputs = self.WhisperEncoder(**whisper_encoder_inputs)

        relation_logits = self.RelationClassifier(encoder_outputs.last_hidden_state)

        relationclass_loss = self.relationclass_loss(relation_logits,whisper_encoder_relation_labels)

        predicted_relations_id_list = self.get_predicted_relations(relation_logits)
        relation_label_id_list = self.get_relation_label_lists(whisper_encoder_relation_labels)

        predicted_relations_type_list = self.relation_label_id_to_type(predicted_relations_id_list)
        relation_label_type_list = self.relation_label_id_to_type(relation_label_id_list)

        decoder_labels_init = whisper_decoder_inputs_train['input_ids']

        decoder_input_ids = self.concat_latent_rel_to_decoder_input(predicted_relations_type_list,decoder_labels_init,stage)
        decoder_input_ids = decoder_input_ids[:,:448]


        decoder_labels = self.shift_tokens_left(decoder_input_ids,self.tokenizer.pad_token_id)
        decoder_labels = decoder_labels[:,:448]
        #decoder
        decoder_outputs = self.WhisperDecoder(
            input_ids=decoder_input_ids,
            attention_mask=(decoder_input_ids != self.WhisperGenModel.config.pad_token_id).long(),
            encoder_hidden_states = encoder_outputs.last_hidden_state
            )
        decoder_outputs_hidden_states = decoder_outputs.last_hidden_state
        #lmhead
        lm_logits = self.WhisperDecoderLmhead(decoder_outputs_hidden_states)
        loss_decoder = self.loss_fct(
            lm_logits.view(-1, len(self.hypernum.processor.tokenizer)), 
            decoder_labels.reshape(-1)
            )
        loss = loss_decoder+relationclass_loss

        self.log("total_loss", loss,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        self.log("relation_loss", relationclass_loss,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        self.log("decoder_loss", loss_decoder,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        return loss
    def validation_step(self, batch,batch_idx):
        self.test_val_step(batch,batch_idx)
    def test_step(self, batch,batch_idx):
        self.test_val_step(batch,batch_idx)
    def test_val_step(self, batch,batch_idx):
        triplets_results_all_views_dic,triplets_labels_all_views_dic = self.inference_step_all_views(batch,batch_idx)
        results_all_orderviews =[]
        labels_all_orderviews = []
        for order_view in triplets_results_all_views_dic.keys():
            triplets_results_all_list = triplets_results_all_views_dic[order_view]
            
            results_str_all_orderview_i = []
            for rao_sample in triplets_results_all_list:
                results_str_all_orderview_i.append(["#$%".join(item) for item in rao_sample])

            triplets_labels_all_list = triplets_labels_all_views_dic[order_view]

            
            for triplets_list,triplets_labels in zip(triplets_results_all_list,triplets_labels_all_list):
                self.TripletF1MerticDic[order_view](triplets_list,triplets_labels)
                self.RelationF1MerticDic[order_view](triplets_list,triplets_labels)
                self.EntityF1MerticDic[order_view](triplets_list,triplets_labels)
            self.log("{}-实体F1".format(order_view),self.EntityF1MerticDic[order_view],on_epoch=True,sync_dist=True)
            self.log("{}-关系F1".format(order_view),self.RelationF1MerticDic[order_view],on_epoch=True,sync_dist=True)
            self.log("{}-三元组F1".format(order_view),self.TripletF1MerticDic[order_view],on_epoch=True,sync_dist=True)
            results_all_orderviews.append(results_str_all_orderview_i)
        vote_reaults_all = self.vote(results_all_orderviews)
        for vote_item,label_item in zip(vote_reaults_all,triplets_labels_all_list):
            self.RelationF1_vote(vote_item,label_item)
            self.EntityF1_vote(vote_item,label_item)
            self.TripletF1_vote(vote_item,label_item)
            self.log("投票后-实体F1".format(order_view),self.EntityF1_vote,on_epoch=True,sync_dist=True)
            self.log("投票后-关系F1".format(order_view),self.RelationF1_vote,on_epoch=True,sync_dist=True)
            self.log("投票后-三元组F1".format(order_view),self.TripletF1_vote,on_epoch=True,sync_dist=True)

        


            
    def vote(self,rtiples_allviews):
        order_nums = len(rtiples_allviews)
        sample_nums = len(rtiples_allviews[0])
        vote_reaults_all = []
        if self.hypernum.vote_threshold <= order_nums:
            vote_threshold = self.hypernum.vote_threshold
        else:
            vote_threshold = self.hypernum.vote_threshold/2

        for sample_nums in range(sample_nums):
            sam_triple_tuples = set()
            triples_results_sample_all_views = []
            for order_id in range(order_nums):
                triples_results_sample_view_i = rtiples_allviews[order_id][sample_nums]
                triples_results_sample_all_views.append(triples_results_sample_view_i)
                sam_triple_tuples.update(triples_results_sample_view_i)
            vote_reaults_sample = [element.split("#$%") for element in sam_triple_tuples if sum(element in t for t in triples_results_sample_all_views)>=vote_threshold]
            vote_reaults_all.append(vote_reaults_sample)

        return vote_reaults_all


    def inference_step_all_views(self,batch,batch_idx):
        triplets_results_all_views_dic = {}
        triplets_labels_all_views_dic = {}
        for order_view,data_view in batch.items():
            s = time.time()
            triplets_list,triplets_labels = self.inference_step_one_view(order_view,data_view)
            e = time.time()
            print("视角{}耗时{}".format(order_view,e-s))
            triplets_results_all_views_dic[order_view] = triplets_list
            triplets_labels_all_views_dic[order_view] = triplets_labels
            
        return triplets_results_all_views_dic,triplets_labels_all_views_dic
    def inference_step_one_view(self,order_view,data_view):
        stage = data_view[5]
        whisper_encoder_inputs,whisper_encoder_relation_labels,whisper_decoder_inputs_train,whisper_decoder_inputs_predict = self.get_whisper_inputdata(data_view)
        
        raw_sen_list = self.tokenizer.batch_decode(data_view[1]['input_ids'])
        
        #encoder
        encoder_outputs = self.WhisperEncoder(**whisper_encoder_inputs)

        relation_logits = self.RelationClassifier(encoder_outputs.last_hidden_state)

        relationclass_loss = self.relationclass_loss(relation_logits,whisper_encoder_relation_labels)

        predicted_relations_id_list = self.get_predicted_relations(relation_logits)
        relation_label_id_list = self.get_relation_label_lists(whisper_encoder_relation_labels)

        predicted_relations_type_list = self.relation_label_id_to_type(predicted_relations_id_list)
        relation_label_type_list = self.relation_label_id_to_type(relation_label_id_list)
        self.EncoderRelationMerticDic[order_view](predicted_relations_type_list,relation_label_type_list)
        self.log("{}-编码器预测关系F1".format(order_view),self.EncoderRelationMerticDic[order_view],on_epoch=True,sync_dist=True)

        decoder_labels_init = whisper_decoder_inputs_train['input_ids']

        decoder_labels = self.concat_latent_rel_to_decoder_input(predicted_relations_type_list,decoder_labels_init,stage)
        decoder_input_ids = decoder_labels
        
        pred_token_ids = self.beam_search(encoder_outputs.last_hidden_state,decoder_input_ids)
        pred_token_strs_list = [[self.tokenizer.convert_ids_to_tokens(ids) for ids in sen] for sen in pred_token_ids.tolist()]
        
        
        
        triplets_list = []
        for pred_token_strs in pred_token_strs_list:
            triplets_sample = self.convert_tree_squeeze_to_triplet(order_view,pred_token_strs)
            triplets_list.append(triplets_sample)

        
    

        triplets_labels = []
        label_token_strs_list = [[self.tokenizer.convert_ids_to_tokens(ids) for ids in sen] for sen in data_view[2]['input_ids'].tolist()]
        for label_tokne_strs in label_token_strs_list:
            triplets_labels_sample = self.convert_tree_squeeze_to_triplet(order_view,label_tokne_strs)
            triplets_labels.append(triplets_labels_sample)

        return triplets_list,triplets_labels
        



    def get_whisper_inputdata(self,batch_data):
        whisper_encoder_inputs = batch_data[0]
        whisper_encoder_relation_labels = batch_data[4]
        whisper_decoder_inputs_train = batch_data[2]
        whisper_decoder_inputs_predict = batch_data[3]
        
        return whisper_encoder_inputs,whisper_encoder_relation_labels,whisper_decoder_inputs_train,whisper_decoder_inputs_predict
    def get_predicted_relations(self,relation_logits, threshold=0.5):
        
        if not torch.is_floating_point(relation_logits):
            relation_logits = relation_logits.float()
         
        probabilities = torch.sigmoid(relation_logits)
        
        mask = probabilities > threshold 
        mask = mask.int()
        
        
        batch_size, label_num = relation_logits.shape
        predicted_labels = []
        for i in range(batch_size):
            
            labels = torch.where(mask[i] == 1)[0].tolist()
            predicted_labels.append(labels)
        
        return predicted_labels
    def get_relation_label_lists(self,true_labels_tensor):

        if not torch.is_floating_point(true_labels_tensor):
            true_labels_tensor = true_labels_tensor.float()
        
        mask = true_labels_tensor == 1  
        
        batch_size, label_num = true_labels_tensor.shape
        true_labels = []
        for i in range(batch_size):
            labels = torch.where(mask[i])[0].tolist()
            true_labels.append(labels)
        
        return true_labels
    def relation_label_id_to_type(self,r_id_list):
        relation_id_to_type = self.hypernum.relation_id_to_type
        mapped_relation_types = [[relation_id_to_type[idr] for idr in relation_list] for relation_list in r_id_list]
        max_length = max(len(relation_list) for relation_list in mapped_relation_types)
        padding_token = '<relation_type_padding_token>'
        padded_relation_types = [
        relation_list + [padding_token] * (max_length - len(relation_list))
        for relation_list in mapped_relation_types
                                ]
        return padded_relation_types
    def concat_latent_rel_to_decoder_input(self,latent_rel,decoder_labels_init,stage):
        bs,_ = decoder_labels_init.shape

        padding_token = "<relation_type_padding_token>"
        latent_rel = [sublist if sublist else [padding_token] for sublist in latent_rel]
        
        max_length = max(len(sublist) for sublist in latent_rel)
        latent_rel = [sublist + [padding_token] * (max_length - len(sublist)) for sublist in latent_rel]



        batchsize,_  = decoder_labels_init.shape
        latent_rel_start_str = [" The sequence of potential relations is:"]*batchsize
        latent_rel_start_tokens = self.hypernum.processor.tokenizer(latent_rel_start_str,return_tensors="pt")['input_ids'][:,:-1].to(self.device)
        latent_rel_tokens = self.hypernum.processor.tokenizer(
            latent_rel,
            return_tensors="pt",
            is_split_into_words = True
            )['input_ids'][:,2:-1].to(self.device)
        if stage:
            if self.hypernum.with_potional_rel_prompt:
                combined_tensor = torch.cat((latent_rel_start_tokens, latent_rel_tokens, decoder_labels_init[:, 2:]), dim=1)
            else:
                combined_tensor = decoder_labels_init
        else:
            if self.hypernum.with_potional_rel_prompt:
                combined_tensor = torch.cat((latent_rel_start_tokens, latent_rel_tokens,decoder_labels_init[:, 2:23]), dim=1)
            else:
                combined_tensor = decoder_labels_init
        return combined_tensor.to(self.device)
    
    def convert_tree_squeeze_to_triplet(self,order_view,pred_token_strs):

        
        pred_sen = ''.join([i  for i in pred_token_strs if i not in ["<|endoftext|>","<|startoftranscript|>",'<|notimestamps|>']])
        pred_sen = pred_sen.replace("Ġ"," ").strip()
        if "The relation triple extraction result is: " in pred_sen:
            pred_sen = pred_sen.split(r"The relation triple extraction result is: ")[1]
        else:
            return []

        triplets_list = []
        node1_element = self.order_map[order_view[0]]
        node1_type = ''
        if order_view[0]!="r":
            node1_all_type_list = self.hypernum.entity_types
        else:
            node1_all_type_list = self.hypernum.relation_types
        
        node2_element = self.order_map[order_view[1]]
        node2_type = ''
        if order_view[1]!="r":
            node2_all_type_list = self.hypernum.entity_types
        else:
            node2_all_type_list = self.hypernum.relation_types
        
        node3_element = self.order_map[order_view[2]]
        node3_type = ''
        if order_view[2]!="r":
            node3_all_type_list = self.hypernum.entity_types
        else:
            node3_all_type_list = self.hypernum.relation_types

        #按照第一级节点来分割所有预测结果
        if node1_element+" " in pred_sen:
            sub_trees_list = [i for i in pred_sen.split(node1_element+" ") if i.strip()]
        else:
            return []#如果没检测到一级节点标识符，就返回空
        
        for sub_tree in sub_trees_list:
            #按照第二级节点来分割所有预测结果
            if node2_element+" " in sub_tree:
                sub_tree_split_node2 = sub_tree.split(node2_element+" ")
            else:
                continue

            node1_context_list = sub_tree_split_node2[0].strip().split(' ')

            if all(node1_context_list) and node1_context_list:#一级节点内容为空，则跳过
               pass
            else:
                continue
            
            #检查一级节点的合规性
            if node1_context_list[0] in node1_all_type_list and sum([1 if i in node1_all_type_list else 0 for i in node1_context_list])==1:
                t_temp_dict = dict(h=[],r=[],t=[])
                t_temp_dict[order_view[0]].append(node1_context_list)

                for sub_tree_node23 in sub_tree_split_node2[1:]:
                    if node3_element+" " in sub_tree_node23:
                        sub_tree_node23_split = sub_tree_node23.split(node3_element+" ")
                    else:
                        continue
                    node2_context_list = sub_tree_node23_split[0].strip().split(' ')

                    #检查二级节点内容形式的合规性
                    if all(node2_context_list) and node2_context_list:
                        pass
                    else:
                        continue

                    if order_view[1]=='r':
                        if len(node2_context_list)==1:
                            pass
                        else:
                            continue
                    else:
                        if len(node2_context_list)>1:
                            pass
                        else:
                            continue

                    #检查二级节点的合规性
                    if node2_context_list[0] in node2_all_type_list and (sum([1 if i in node2_all_type_list else 0 for i in node2_context_list])==1):
                        if t_temp_dict[order_view[1]]:#如果遇到第二个23node的子树，若合规则保存三元组，然后清空23node缓存
                            if all(t_temp_dict[key] and all(sublist for sublist in t_temp_dict[key]) for key in t_temp_dict):
                                nest_list_v_key = "t"
                                for t_temp_k,t_temp_v in t_temp_dict.items():
                                    if len(t_temp_v)>1:
                                        nest_list_v_key = t_temp_k
                                non_nest_orders = ["h","r","t"]
                                non_nest_orders.remove(nest_list_v_key)
                                for nest_list_item in t_temp_dict[nest_list_v_key]:
                                    temp_t = ['','','','','']
                                    if nest_list_v_key =="r":
                                        temp_t[self.order_to_id_contain_e[nest_list_v_key][0]] = nest_list_item[0]
                                    else:
                                        temp_t[self.order_to_id_contain_e[nest_list_v_key][0]] = " ".join(nest_list_item[1:])
                                        temp_t[self.order_to_id_contain_e[nest_list_v_key][1]] = nest_list_item[0]
                                    
                                    for non_nest_order in non_nest_orders:
                                        if non_nest_order == "r":
                                            temp_t[self.order_to_id_contain_e[non_nest_order][0]] = t_temp_dict[non_nest_order][0][0]
                                        else:
                                            temp_t[self.order_to_id_contain_e[non_nest_order][0]] = " ".join(t_temp_dict[non_nest_order][0][1:])
                                            temp_t[self.order_to_id_contain_e[non_nest_order][1]] = t_temp_dict[non_nest_order][0][0]
                                    triplets_list.append(temp_t)       
                            t_temp_dict[order_view[1]] = []#清空二级节点之后的缓存
                            t_temp_dict[order_view[2]] = []
                        
                        
                        t_temp_dict[order_view[1]].append(node2_context_list)

                        for sub_tree_node3 in sub_tree_node23_split[1:]:
                            
                            node3_context_list = sub_tree_node3.strip().split(' ')

                            if all(node3_context_list) and node3_context_list:
                                pass
                            else:
                                continue

                            if order_view[2]=='r':
                                if len(node3_context_list)==1:
                                    pass
                                else:
                                    continue
                            else:
                                if len(node3_context_list)>1:
                                    pass
                                else:
                                    continue



                            #检查node3是否合规
                            if node3_context_list[0] in node3_all_type_list and (sum([1 if i in node3_all_type_list else 0 for i in node3_context_list])==1):
                                t_temp_dict[order_view[2]].append(node3_context_list)
                            else:
                                continue#如果节点三不合规，则跳过
                    else:
                        continue#二级节点不合规，这个二级子树舍弃
            else:
                continue#一级节点不合规，这个三级子树舍弃
            
            if all(t_temp_dict[key] and all(sublist for sublist in t_temp_dict[key]) for key in t_temp_dict):#retraced数据集中有数据被截断，导致生成的关系三元组数量不全，直接丢弃
                nest_list_v_key = "t"
                for t_temp_k,t_temp_v in t_temp_dict.items():
                    if len(t_temp_v)>1:
                        nest_list_v_key = t_temp_k
                non_nest_orders = ["h","r","t"]
                non_nest_orders.remove(nest_list_v_key)
                for nest_list_item in t_temp_dict[nest_list_v_key]:
                    temp_t = ['','','','','']
                    if nest_list_v_key =="r":
                        temp_t[self.order_to_id_contain_e[nest_list_v_key][0]] = nest_list_item[0]
                    else:
                        temp_t[self.order_to_id_contain_e[nest_list_v_key][0]] = " ".join(nest_list_item[1:])
                        temp_t[self.order_to_id_contain_e[nest_list_v_key][1]] = nest_list_item[0]
                    
                    for non_nest_order in non_nest_orders:
                        if non_nest_order == "r":
                            temp_t[self.order_to_id_contain_e[non_nest_order][0]] = t_temp_dict[non_nest_order][0][0]
                        else:
                            temp_t[self.order_to_id_contain_e[non_nest_order][0]] = " ".join(t_temp_dict[non_nest_order][0][1:])
                            temp_t[self.order_to_id_contain_e[non_nest_order][1]] = t_temp_dict[non_nest_order][0][0]
                    triplets_list.append(temp_t)
        triplets_list = [triplet for triplet in triplets_list if not any(s == '' or s.strip() == '' for s in triplet)]
                
        return triplets_list
    def convert_orderview_to_token(self,order_view_str):
        

        order_tokens = [self.order_map[order_str] for order_str in order_view_str]

        return order_tokens
    def beam_search(self,
                    encoder_outputs,
                    decoder_input_ids_init
                    ):
        eos_token_id = self.tokenizer.eos_token_id
        num_beams = self.hypernum.num_beams
        max_length = self.hypernum.decoder_max_length
        bs,_,_ = encoder_outputs.shape
        bms = num_beams
        rns = bs * bms

        max_length = max_length-decoder_input_ids_init.shape[1]

        decoder_input_ids = decoder_input_ids_init
        # init hypotheses, scores and flags
        hyps = decoder_input_ids[:,-1:]  # (bs, 1)
        hyps = hyps.unsqueeze(1).repeat(1, bms, 1).view(rns, -1)  # (rns, 1), the hypothesis of current beam

        

        decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, bms, 1).view(rns, -1)

        enc_out = encoder_outputs.unsqueeze(1).repeat(1, bms, 1, 1).view(rns, encoder_outputs.size(-2), encoder_outputs.size(-1))

        scores = torch.zeros(bms).float().to(self.device)
        scores[1:] = float("-inf")
        scores = scores.repeat(bs, 1).view(rns)                     # (rns,), the scores of current beam
        end_flag = torch.zeros(rns).bool().to(self.device)                        # (rns,), whether current beam is finished
        # Initialize cache at the start of decoding
        cache = {}
        for i in range(max_length):
            #print(i+int(decoder_input_ids_init.shape[1]))


            if end_flag.view(bs,bms).sum(-1).all():#最大概率的beam停止就该停止了
                break
            if i==0:
                decoder_outputs = self.WhisperDecoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=enc_out,
                        use_cache=True
                    )
                logits = self.WhisperDecoderLmhead(decoder_outputs.last_hidden_state)
                logits = logits[:, -1]
                logp = F.log_softmax(logits, dim=-1)
                cache = decoder_outputs.past_key_values
            else:
                decoder_input_ids = torch.cat([decoder_input_ids,hyps[:,-1:]],dim=-1)
                decoder_outputs = self.WhisperDecoder(
                        input_ids=decoder_input_ids[:,-1:],
                        encoder_hidden_states=enc_out,
                        use_cache=True,
                        return_dict=True,
                        past_key_values = cache
                    )
                cache = decoder_outputs.past_key_values
                logits = self.WhisperDecoderLmhead(decoder_outputs.last_hidden_state)
                logits = logits[:, -1]
                logp = F.log_softmax(logits, dim=-1)
                
            # local pruning: prune non-topk scores
            topk_logp, topk_idxs = logp.topk(k=bms, dim=-1)  # (rns, vocab) -> (rns, bms)
            topk_logp, topk_idxs = topk_logp.to(self.device), topk_idxs.to(self.device)
            # masked finished beams
            topk_logp = self.mask_finished_scores(topk_logp, end_flag)
            topk_idxs = self.mask_finished_preds(topk_idxs, end_flag, eos_token_id)
            
            # calculate scores of new beams
            scores = scores.view(rns, 1)
            scores = scores + topk_logp  # (rns, 1) + (rns, bms) -> (rns, bms)
            scores = scores.view(bs, bms * bms)
            # global pruning
            scores, offset_k_idxs = scores.topk(k=bms, dim=-1)  # (bs, bms)
            scores = scores.view(rns, 1)
            offset_k_idxs = offset_k_idxs.view(-1)

            # calculate the predicted token at current decoding step
            base_k_idxs = torch.arange(bs, device=scores.device) * bms * bms
            # wrong implementation:
            # base_k_idxs = base_k_idxs.repeat(bms).view(-1)
            # correct implementation:
            base_k_idxs = base_k_idxs.unsqueeze(-1).repeat(1, bms).view(-1)
            # e.g. base_k_idxs: (0, 0, 0, 9, 9, 9, 81, 81, 81)
            best_k_idxs = base_k_idxs + offset_k_idxs.view(-1)
            best_k_pred = torch.index_select(topk_idxs.view(-1), dim=-1, index=best_k_idxs)

            # retrive the old hypotheses of best k beams
            best_hyp_idxs = best_k_idxs.div(bms, rounding_mode="floor")
            last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyp_idxs)  # (rns, i)

            # concat the old hypotheses with the new predicted token
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)  # (rns, i)

            # refresh end_flag
            end_flag = torch.eq(hyps[:, -1], eos_token_id).view(-1)
        
        # get the best hyp
        scores = scores.view(-1, bms)  # (rns, bms)
        _, best_hyp_idxs = scores.topk(k=1, dim=-1)  # (bs, 1)
        best_hyp_idxs = best_hyp_idxs.view(-1)
        idxs = torch.arange(bs, device=scores.device) * bms
        idxs = idxs.unsqueeze(1).repeat(1, 1).view(-1)
        best_hyp_idxs += idxs
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyp_idxs.to(self.device))

        best_hyps = torch.cat([decoder_input_ids_init,best_hyps],dim=-1)
        pred_tokens = best_hyps

        return pred_tokens
    def mask_finished_scores(self,scores, end_flag, inf=-float("inf")):
        """
        Example of end_flag:
            0
            1
            0
            1
            1
        Corresponding mask `mask_to_inf`:
            0 0 0 0 0
            0 1 1 1 1
            0 0 0 0 0
            0 1 1 1 1
            0 1 1 1 1
        Corresponding mask `mask_to_zero`:
            0 0 0 0 0
            1 0 0 0 0
            0 0 0 0 0
            1 0 0 0 0
            1 0 0 0 0
        In the above case, there're five samples and five beams.
        The second and the fivth samples have mask_to_zero beam searching.

        """
        rns, bms = scores.size()
        #assert end_flag.size(0) == rns and end_flag.ndim == 1
        zero_mask = scores.new_zeros(rns, 1)
        mask_to_zero = torch.cat([end_flag.view(rns, 1), zero_mask.repeat(1, bms - 1)], dim=-1)  # (rns, bms)
        mask_to_inf = torch.cat([zero_mask, end_flag.view(rns, 1).repeat(1, bms - 1)], dim=-1)  # (rns, bms)
        scores = scores.masked_fill(mask_to_zero.bool(), 0.)
        scores = scores.masked_fill(mask_to_inf.bool(), inf)
        return scores

    def mask_finished_preds(self,preds, end_flag, eos_id):
        # Force preds to be all `sos` for finished beams.
        rns, bms = preds.size()
        finished = end_flag.view(-1, 1).repeat(1, bms)  # (rns, bms)
        preds.masked_fill_(finished.bool(), eos_id)
        return preds
    def shift_tokens_left(self,input_ids: torch.Tensor, pad_token_id: int):
        """
        Shift input ids one token to the left.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()  # 向左移动
        shifted_input_ids[:, -1] = pad_token_id  # 在末尾填充 pad_token_id

        if pad_token_id is None:
            raise ValueError("pad_token_id must be provided.")

        # 将可能的 -100 值替换为 pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids.to(self.device)

if __name__ == "__main__":
    class Hypernum:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    hyper_dic = dict(
        data_path = r"Datasets/speech_ReTracred",
        audio_file = r"/root/autodl-tmp/speech_conll04/audio",
        whisper_model_dir = r"PretrainedSpeechModel/whisper-base.en",
        data_type = "train",
        train_order_views = [0,1,2,3,4,5],
        predict_order_views = [0,1,2,3,4,5],
        del_no_relation_type = True,
        batch_size = 4,
    )
    hypernum = Hypernum(**hyper_dic)
    datamodule = SpeechReDatamodule(hypernum=hypernum)
    model = SpeechReModel(hypernum)