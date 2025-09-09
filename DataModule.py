import os
import torch
import random
import librosa
import json
import numpy as np
import lightning as L
from torch.utils.data import ConcatDataset,StackDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import WhisperProcessor,\
                        WhisperFeatureExtractor,\
                        WhisperTokenizer

class SpeechRE_Dataset(Dataset):
    def __init__(self, data_path,audio_file, data_type,order_view,del_no_relation_type,hypernum=None):
        super().__init__()
        self.data_path = data_path
        self.audio_file = audio_file
        self.data_type = data_type
        self.order_view = order_view
        self.del_no_relation_type = del_no_relation_type
        self.all_data,entities_types,relations_types = self.get_alldata()

        self.order_map = dict(h = '<head_entity>',r = '<relation_type>',t = '<tail_entity>')
        self.order_to_id = {'<head_entity>':0,'<relation_type>':1,'<tail_entity>':2}
        self.id_to_order = {0:'<head_entity>',1:'<relation_type>',2:'<tail_entity>'}
        self.order_to_id_contain_e = dict(h = [0,1],r=[2],t = [3,4])
        self.hypernum = hypernum

    def __getitem__(self, index):
        data_item  = self.all_data[index]
        sentence = data_item["sentence"]
        target,decoder_input_text,entity_list,relation_list = self.get_target(data_item,self.order_view)

        audio_id = data_item["audio_id"]
        if "conll04" in self.data_path.lower() or "retracred" in self.data_path.lower():
            speechfiledir = os.path.join(self.audio_file, audio_id+".wav")
        else:
            if "tts" in self.audio_file:
                speechfiledir = os.path.join(self.audio_file, "tts_"+audio_id+".wav")
            else:
                speechfiledir = os.path.join(self.audio_file, audio_id+".npy")
        if ".wav" in speechfiledir:
            try:
                waveform, sampling_rate = librosa.load(speechfiledir, sr=16000)
            except Exception as e:
                target = "There is no audio input."
                waveform = np.zeros(500)  # 长度为 500 的全零序列
                sampling_rate = 16000  # 设置采样率为 16000 Hz
        else:
            waveform = np.load(speechfiledir)
            sampling_rate = 16000

        
        return waveform,sentence,target,decoder_input_text,entity_list,relation_list,self.data_type
    def get_alldata(self):
        data_file_path = os.path.join(self.data_path, self.data_type + ".json")
        dataset_info_file_path = os.path.join(self.data_path,"dataset_info.json")

        with open(data_file_path, "r",encoding='utf-8') as f:
            all_data = json.load(f)
        with open(dataset_info_file_path, "r",encoding='utf-8') as f:
            dataset_info = json.load(f)
        entities_types = entities_types = [e_item if e_item.startswith("<") else '<'+e_item+'>' for e_item in dataset_info["entities_type"]]

        relations_types = [r_item if r_item.startswith("<") else '<'+r_item+'>' for r_item in dataset_info["relations_type"]]
        
        return all_data,entities_types,relations_types
    def __len__(self):
        return len(self.all_data)
    
    def get_target(self,data_item,order_view):
        if order_view=="random":
            order_view = random.choice(["hrt","htr","rth","rht","trh","thr"])
            self.order_view = order_view

        if order_view=="hrt":
            target,decoder_input_text,entity_list,relation_list = self.get_target_hrt(data_item)
        if order_view=="htr":
            target,decoder_input_text,entity_list,relation_list = self.get_target_htr(data_item)
        if order_view=="rht":
            target,decoder_input_text,entity_list,relation_list = self.get_target_rht(data_item)
        if order_view=="rth":
            target,decoder_input_text,entity_list,relation_list = self.get_target_rth(data_item)
        if order_view=="trh":
            target,decoder_input_text,entity_list,relation_list = self.get_target_trh(data_item)
        if order_view=="thr":
            target,decoder_input_text,entity_list,relation_list = self.get_target_thr(data_item)
        if order_view=="entity":
            target,decoder_input_text,entity_list,relation_list = self.get_target_entity(data_item)
        return target,decoder_input_text,entity_list,relation_list
    def get_entity_relation_list(self,data_item):
        entity_list = []
        relation_list = []
        for anotation in data_item["data_annotation"]:
            if anotation['type'] == "entity":
                entity_list.append((int(anotation['id']),
                                    anotation['value']['start'],
                                    anotation['value']['end'],
                                    anotation['value']['text'],
                                    anotation['value']['labels']
                                    ))
            if anotation['type'] == "relation":
                for r_item in anotation["labels"]:
                    if self.del_no_relation_type:
                        if r_item != "no_relation":
                            relation_list.append((
                                                anotation['from_id'],
                                                anotation['to_id'],
                                                r_item
                                                ))
                    else:
                        relation_list.append((
                                            anotation['from_id'],
                                            anotation['to_id'],
                                            r_item
                                            ))
        return entity_list,relation_list
    def get_target_hrt(self,data_item):
        entity_list,relation_list = self.get_entity_relation_list(data_item)
        son1_nodes = []
        son23_nodes = []
        for r_item in relation_list:
            if r_item[0] not in son1_nodes:
                son1_nodes.append(r_item[0])
                son23_nodes.append([ [r_item[2],[ r_item[1] ] ]])
            else:
                exist_son1_index = son1_nodes.index(r_item[0])
                exist_son23_node = son23_nodes[exist_son1_index]
                node2_exist_flag = False
                for node2_item_index in range(len(exist_son23_node)):
                    if exist_son23_node[node2_item_index][0]==r_item[2]:
                        son23_nodes[exist_son1_index][node2_item_index][1].append(r_item[1])
                        node2_exist_flag = True
                if not node2_exist_flag:
                    son23_nodes[exist_son1_index].append([r_item[2],[r_item[1]]])
        assert len(son1_nodes)==len(son23_nodes),"父节点和子节点数量不一致！"
        target = ''
        for node1_item,node23_items in zip(son1_nodes,son23_nodes):
            assert node1_item==entity_list[node1_item][0],"节点编号不一致！"
            node1_context = entity_list[node1_item][3]
            node1_type = entity_list[node1_item][4]
            
            if not node1_type.startswith('<'):
                node1_type = '<'+node1_type+'>'
            target+=" <head_entity> "+node1_type+" "+ node1_context
            for node23_item in node23_items:
                node2_type = node23_item[0]
                if not node2_type.startswith('<'):
                    node2_type = '<'+node2_type+'>'

                target+= " <relation_type> "+node2_type
                node3_items = node23_item[1]
                for node3_item in node3_items:
                    node3_context = entity_list[node3_item][3]
                    node3_type = entity_list[node3_item][4]
                    if not node3_type.startswith('<'):
                        node3_type = '<'+node3_type+'>'
                    target+=" <tail_entity> " + node3_type+" "+ node3_context
        if len(target)==0:
            target="No relation triples were detected in the input data."
        decoder_input_text = self.get_order_control_token_squeeze()+"The relation triple extraction result is: "
        target = decoder_input_text + target
        return target,decoder_input_text,entity_list,relation_list
    def get_target_htr(self,data_item):
        entity_list,relation_list = self.get_entity_relation_list(data_item)
        son1_nodes = []
        son23_nodes = []
        for r_item in relation_list:
            if r_item[0] not in son1_nodes:
                son1_nodes.append(r_item[0])
                son23_nodes.append([ [r_item[1],[ r_item[2] ] ]])
            else:
                exist_son1_index = son1_nodes.index(r_item[0])
                exist_son23_node = son23_nodes[exist_son1_index]
                node2_exist_flag = False
                for node2_item_index in range(len(exist_son23_node)):
                    if exist_son23_node[node2_item_index][0]==r_item[1]:
                        son23_nodes[exist_son1_index][node2_item_index][1].append(r_item[2])
                        node2_exist_flag = True
                if not node2_exist_flag:
                    son23_nodes[exist_son1_index].append([r_item[1],[r_item[2]]])
        assert len(son1_nodes)==len(son23_nodes),"父节点和子节点数量不一致！"
        target = ''
        for node1_item,node23_items in zip(son1_nodes,son23_nodes):
            assert node1_item==entity_list[node1_item][0],"节点编号不一致！"
            node1_context = entity_list[node1_item][3]
            node1_type = entity_list[node1_item][4]
            
            if not node1_type.startswith('<'):
                node1_type = '<'+node1_type+'>'
            target+=" <head_entity> "+node1_type+" "+ node1_context
            for node23_item in node23_items:
                

                node2_context = entity_list[node23_item[0]][3]
                node2_type = entity_list[node23_item[0]][4]
                if not node2_type.startswith('<'):
                    node2_type = '<'+node2_type+'>'

                target+= " <tail_entity> "+node2_type+" "+ node2_context
                node3_items = node23_item[1]
                for node3_item in node3_items:
                    
                    node3_type = node3_item
                    if not node3_type.startswith('<'):
                        node3_type = '<'+node3_type+'>'
                    target+=" <relation_type> " + node3_type
        if len(target)==0:
            target="No relation triples were detected in the input data."
        decoder_input_text = self.get_order_control_token_squeeze()+"The relation triple extraction result is: "
        target = decoder_input_text + target
        return target,decoder_input_text,entity_list,relation_list
    def get_target_rht(self,data_item):
        entity_list,relation_list = self.get_entity_relation_list(data_item)
        son1_nodes = []
        son23_nodes = []
        for r_item in relation_list:
            if r_item[2] not in son1_nodes:
                son1_nodes.append(r_item[2])
                son23_nodes.append([ [r_item[0],[ r_item[1] ] ]])
            else:
                exist_son1_index = son1_nodes.index(r_item[2])
                exist_son23_node = son23_nodes[exist_son1_index]
                node2_exist_flag = False
                for node2_item_index in range(len(exist_son23_node)):
                    if exist_son23_node[node2_item_index][0]==r_item[0]:
                        son23_nodes[exist_son1_index][node2_item_index][1].append(r_item[1])
                        node2_exist_flag = True
                if not node2_exist_flag:
                    son23_nodes[exist_son1_index].append([r_item[0],[r_item[1]]])
        assert len(son1_nodes)==len(son23_nodes),"父节点和子节点数量不一致！"
        target = ''
        for node1_item,node23_items in zip(son1_nodes,son23_nodes):
            
            node1_type = node1_item
            
            if not node1_type.startswith('<'):
                node1_type = '<'+node1_type+'>'
            target+=" <relation_type> "+node1_type
            for node23_item in node23_items:
                

                node2_context = entity_list[node23_item[0]][3]
                node2_type = entity_list[node23_item[0]][4]
                if not node2_type.startswith('<'):
                    node2_type = '<'+node2_type+'>'

                target+= " <head_entity> "+node2_type+" "+ node2_context
                node3_items = node23_item[1]
                for node3_item in node3_items:
                    node3_context = entity_list[node3_item][3]
                    node3_type = entity_list[node3_item][4]
                    if not node3_type.startswith('<'):
                        node3_type = '<'+node3_type+'>'
                    target+=" <tail_entity> " + node3_type+" "+ node3_context
        if len(target)==0:
            target="No relation triples were detected in the input data."
        decoder_input_text = self.get_order_control_token_squeeze()+"The relation triple extraction result is: "
        target = decoder_input_text + target
        return target,decoder_input_text,entity_list,relation_list
    def get_target_rth(self,data_item):
        entity_list,relation_list = self.get_entity_relation_list(data_item)
        son1_nodes = []
        son23_nodes = []
        for r_item in relation_list:
            if r_item[2] not in son1_nodes:
                son1_nodes.append(r_item[2])
                son23_nodes.append([ [r_item[1],[ r_item[0] ] ]])
            else:
                exist_son1_index = son1_nodes.index(r_item[2])
                exist_son23_node = son23_nodes[exist_son1_index]
                node2_exist_flag = False
                for node2_item_index in range(len(exist_son23_node)):
                    if exist_son23_node[node2_item_index][0]==r_item[1]:
                        son23_nodes[exist_son1_index][node2_item_index][1].append(r_item[0])
                        node2_exist_flag = True
                if not node2_exist_flag:
                    son23_nodes[exist_son1_index].append([r_item[1],[r_item[0]]])
        assert len(son1_nodes)==len(son23_nodes),"父节点和子节点数量不一致！"
        target = ''
        for node1_item,node23_items in zip(son1_nodes,son23_nodes):
            
            node1_type = node1_item
            
            if not node1_type.startswith('<'):
                node1_type = '<'+node1_type+'>'
            target+=" <relation_type> "+node1_type
            for node23_item in node23_items:
                

                node2_context = entity_list[node23_item[0]][3]
                node2_type = entity_list[node23_item[0]][4]
                if not node2_type.startswith('<'):
                    node2_type = '<'+node2_type+'>'

                target+= " <tail_entity> "+node2_type+" "+ node2_context
                node3_items = node23_item[1]
                for node3_item in node3_items:
                    node3_context = entity_list[node3_item][3]
                    node3_type = entity_list[node3_item][4]
                    if not node3_type.startswith('<'):
                        node3_type = '<'+node3_type+'>'
                    target+=" <head_entity> " + node3_type+" "+ node3_context
        if len(target)==0:
            target="No relation triples were detected in the input data."
        decoder_input_text = self.get_order_control_token_squeeze()+"The relation triple extraction result is: "
        target = decoder_input_text + target
        return target,decoder_input_text,entity_list,relation_list
    def get_target_trh(self,data_item):
        entity_list,relation_list = self.get_entity_relation_list(data_item)
        son1_nodes = []
        son23_nodes = []
        for r_item in relation_list:
            if r_item[1] not in son1_nodes:
                son1_nodes.append(r_item[1])
                son23_nodes.append([ [r_item[2],[ r_item[0] ] ]])
            else:
                exist_son1_index = son1_nodes.index(r_item[1])
                exist_son23_node = son23_nodes[exist_son1_index]
                node2_exist_flag = False
                for node2_item_index in range(len(exist_son23_node)):
                    if exist_son23_node[node2_item_index][0]==r_item[2]:
                        son23_nodes[exist_son1_index][node2_item_index][1].append(r_item[0])
                        node2_exist_flag = True
                if not node2_exist_flag:
                    son23_nodes[exist_son1_index].append([r_item[2],[r_item[0]]])
        assert len(son1_nodes)==len(son23_nodes),"父节点和子节点数量不一致！"
        target = ''
        for node1_item,node23_items in zip(son1_nodes,son23_nodes):
            assert node1_item==entity_list[node1_item][0],"节点编号不一致！"
            node1_context = entity_list[node1_item][3]
            node1_type = entity_list[node1_item][4]
            
            if not node1_type.startswith('<'):
                node1_type = '<'+node1_type+'>'
            target+=" <tail_entity> "+node1_type+" "+ node1_context
            for node23_item in node23_items:
                node2_type = node23_item[0]
                if not node2_type.startswith('<'):
                    node2_type = '<'+node2_type+'>'

                target+= " <relation_type> "+node2_type
                node3_items = node23_item[1]
                for node3_item in node3_items:
                    node3_context = entity_list[node3_item][3]
                    node3_type = entity_list[node3_item][4]
                    if not node3_type.startswith('<'):
                        node3_type = '<'+node3_type+'>'
                    target+=" <head_entity> " + node3_type+" "+ node3_context
        if len(target)==0:
            target="No relation triples were detected in the input data."
        decoder_input_text = self.get_order_control_token_squeeze()+"The relation triple extraction result is: "
        target = decoder_input_text + target
        return target,decoder_input_text,entity_list,relation_list
    def get_target_thr(self,data_item):
        entity_list,relation_list = self.get_entity_relation_list(data_item)
        son1_nodes = []
        son23_nodes = []
        for r_item in relation_list:
            if r_item[1] not in son1_nodes:
                son1_nodes.append(r_item[1])
                son23_nodes.append([ [r_item[0],[ r_item[2] ] ]])
            else:
                exist_son1_index = son1_nodes.index(r_item[1])
                exist_son23_node = son23_nodes[exist_son1_index]
                node2_exist_flag = False
                for node2_item_index in range(len(exist_son23_node)):
                    if exist_son23_node[node2_item_index][0]==r_item[0]:
                        son23_nodes[exist_son1_index][node2_item_index][1].append(r_item[2])
                        node2_exist_flag = True
                if not node2_exist_flag:
                    son23_nodes[exist_son1_index].append([r_item[0],[r_item[2]]])
        assert len(son1_nodes)==len(son23_nodes),"父节点和子节点数量不一致！"
        target = ''
        for node1_item,node23_items in zip(son1_nodes,son23_nodes):
            assert node1_item==entity_list[node1_item][0],"节点编号不一致！"
            node1_context = entity_list[node1_item][3]
            node1_type = entity_list[node1_item][4]
            
            if not node1_type.startswith('<'):
                node1_type = '<'+node1_type+'>'
            target+=" <tail_entity> "+node1_type+" "+ node1_context
            for node23_item in node23_items:
                

                node2_context = entity_list[node23_item[0]][3]
                node2_type = entity_list[node23_item[0]][4]
                if not node2_type.startswith('<'):
                    node2_type = '<'+node2_type+'>'

                target+= " <head_entity> "+node2_type+" "+ node2_context
                node3_items = node23_item[1]
                for node3_item in node3_items:
                    
                    node3_type = node3_item
                    if not node3_type.startswith('<'):
                        node3_type = '<'+node3_type+'>'
                    target+=" <relation_type> " + node3_type
        if len(target)==0:
            target="No relation triples were detected in the input data."
        decoder_input_text = self.get_order_control_token_squeeze()+"The relation triple extraction result is: "
        target = decoder_input_text + target
        return target,decoder_input_text,entity_list,relation_list
    def get_target_entity(self,data_item):
        entity_list,relation_list = self.get_entity_relation_list(data_item)
        target = ""
        for entity_item in entity_list:
            target += entity_item[4]+" "+entity_item[3]+" "
        if len(target)==0:
            target = "No entities were detected in the input data."

        decoder_input_text = "The entity extraction result is: "
        target = decoder_input_text + target
        return target,decoder_input_text,entity_list,relation_list
    def get_order_control_token_squeeze(self):
        assert self.order_view!="entity","实体生成任务没有顺序控制token"
        order_control_token_map = dict(
            h = "<head_entity>",
            t = "<tail_entity>",
            r = "<relation_type>"
        )
        order_control_token_squeeze = " ".join([order_control_token_map[item] for item in self.order_view])
        order_control_token_squeeze = " The order of relation generation is: " + order_control_token_squeeze +" "
        return order_control_token_squeeze
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
                        if t_temp_dict[order_view[1]]:#如果遇到第二个23node的子树，保存三元组，然后清空23node缓存
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
            
            if all(t_temp_dict.values()):#retraced数据集中有数据被截断，导致生成的关系三元组数量不全，直接丢弃
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
class SpeechReDatamodule(L.LightningDataModule):
    def __init__(self,hypernum):
        super().__init__()
        self.hypernum = hypernum
        tokenizer_kwargs = {
            "additional_special_tokens": ['<head_entity>', '<tail_entity>', '<relation_type>'],
        }  
        self.processor = WhisperProcessor.from_pretrained(self.hypernum.whisper_model_dir,
                                                          **tokenizer_kwargs)
        self.entity_types,self.relation_types,self.relation_type_to_id,self.relation_id_to_type = self.get_entity_relation_types()
        self.processor.tokenizer.add_tokens(self.entity_types,special_tokens = True)
        self.processor.tokenizer.add_tokens(self.relation_types,special_tokens = True)
        self.hypernum.processor = self.processor
        self.hypernum.relation_type_to_id = self.relation_type_to_id
        self.hypernum.relation_id_to_type = self.relation_id_to_type
        self.hypernum.entity_types = self.entity_types
        self.hypernum.relation_types = self.relation_types

        self.train_orders,self.predict_orders = self.get_order_views()

    def get_entity_relation_types(self):
        dataset_info_file_dir = os.path.join(self.hypernum.data_path,'dataset_info.json')
        with open(dataset_info_file_dir,'r',encoding='utf-8') as f:
            dataset_info_dic = json.load(f)
        entity_types = [item if item.startswith('<') else "<{}>".format(item) for item in dataset_info_dic["entities_type"]]
        relation_types = [item if item.startswith('<') else "<{}>".format(item) for item in dataset_info_dic["relations_type"]]
        relation_types.append("<relation_type_padding_token>")
        relation_type_to_id = {relation: idx for idx, relation in enumerate(relation_types)}
        relation_id_to_type = {idx: relation for idx, relation in enumerate(relation_types)}
        return entity_types,relation_types,relation_type_to_id,relation_id_to_type
    def get_order_views(self):
        order_map = {0:"hrt",1:"htr",2:"rht",3:"rth",4:"trh",5:"thr"}
        train_order_views_after_map = [order_map[i] for i in self.hypernum.train_order_views]
        predict_order_views_after_map = [order_map[i] for i in self.hypernum.predict_order_views]
        return train_order_views_after_map,predict_order_views_after_map
    def setup(self, stage):
        if stage in ["fit","validate"]:
            if self.hypernum.train_data_mix == 'random':
                self.train_dataset_concat = SpeechRE_Dataset(
                        self.hypernum.data_path,
                        self.hypernum.audio_file,
                        "train",
                        'random',
                        self.hypernum.del_no_relation_type
                                                )
            else:
                train_dataset_list = []
                for train_order in self.train_orders:
                    train_dataset_list.append(
                        SpeechRE_Dataset(
                        self.hypernum.data_path,
                        self.hypernum.audio_file,
                        "train",
                        train_order,
                        self.hypernum.del_no_relation_type
                                                )
                    )
                if self.hypernum.add_entity_target:
                    train_dataset_list.append(
                        SpeechRE_Dataset(
                        self.hypernum.data_path,
                        self.hypernum.audio_file,
                        "train",
                        'entity',
                        self.hypernum.del_no_relation_type
                                                )
                    )
                self.train_dataset_concat = ConcatDataset(train_dataset_list)

            self.dev_dataset_dict = {}
            for predict_order in self.predict_orders:
                self.dev_dataset_dict[predict_order] = SpeechRE_Dataset(
                                                    self.hypernum.data_path,
                                                    self.hypernum.audio_file,
                                                    "dev",
                                                    predict_order,
                                                    self.hypernum.del_no_relation_type
                                              )
        else:
            self.test_dataset_dict = {}
            for predict_order in self.predict_orders:
                self.test_dataset_dict[predict_order] = SpeechRE_Dataset(
                                                    self.hypernum.data_path,
                                                    self.hypernum.audio_file,
                                                    "test",
                                                    predict_order,
                                                    self.hypernum.del_no_relation_type
                                              )

            
    def train_dataloader(self):
        train_dataloader = DataLoader(
                                    self.train_dataset_concat,
                                    batch_size=self.hypernum.batch_size,
                                    shuffle=True,#训练阶段开启随机采样
                                    collate_fn=self.collate_train,
                                    num_workers = self.hypernum.dataloader_numworkers
                                        )
        return train_dataloader
    
    def val_dataloader(self):
        dev_datasets_stack = StackDataset(**self.dev_dataset_dict)
        dev_dataloader_stack = DataLoader(
                                dev_datasets_stack,
                                batch_size=self.hypernum.batch_size,
                                collate_fn=self.collate_infer,
                                num_workers = self.hypernum.dataloader_numworkers
                                    )
        
        return dev_dataloader_stack
    
    def test_dataloader(self):
        test_datasets_stack = StackDataset(**self.test_dataset_dict)
        test_dataloader_stack = DataLoader(
                                test_datasets_stack,
                                batch_size=self.hypernum.batch_size,
                                collate_fn=self.collate_infer,
                                num_workers = self.hypernum.dataloader_numworkers
                                    )
        return test_dataloader_stack
    def collate_train(self,data):
        #waveform,sentence,target,decoder_input_text,entity_list,relation_list
        waveforms = [data_item[0] for data_item in data]
        sentences = [data_item[1] for data_item in data]
        target = [data_item[2] for data_item in data]
        decoder_input_texts = [data_item[3] for data_item in data]
        entity_list_batch = [data_item[4] for data_item in data]
        relation_list_batch = [data_item[5] for data_item in data]
        stage = data[0][6]
        triplet_label_list = []
        for entity_list,relation_list in zip(entity_list_batch,relation_list_batch):
            t_temp = []
            for r in relation_list:
                h_e = entity_list[r[0]]
                t_e = entity_list[r[1]]
                r_type = r[2] if r[2].startswith('<') else '<{}>'.format(r[2])
                h_e_text = h_e[3] 
                h_e_type = h_e[4] if h_e[4].startswith('<') else '<{}>'.format(h_e[4])
                t_e_text = t_e[3] 
                t_e_type = t_e[4] if t_e[4].startswith('<') else '<{}>'.format(t_e[4])
                t_temp.append([h_e_text,h_e_type,r_type,t_e_text,t_e_type])
            triplet_label_list.append(t_temp)
        


        input_relation_type_ids_temp = []
        for data_item in data:
            relaton_type_sample = []
            
            for r_item in data_item[5]:
                r_temp = r_item[2]
                r_temp = r_temp if r_temp.startswith('<') else "<{}>".format(r_temp)
                if self.hypernum.relation_type_to_id[r_temp] not in relaton_type_sample:
                    relaton_type_sample.append(self.hypernum.relation_type_to_id[r_temp])
            input_relation_type_ids_temp.append(relaton_type_sample)
        input_relation_type_ids = []
        for seq in input_relation_type_ids_temp:
            tensor_temp = torch.zeros(len(self.hypernum.relation_type_to_id)-1)
            for pos_relation_ind in seq:
                tensor_temp[pos_relation_ind] = 1
            input_relation_type_ids.append(tensor_temp)
        input_relation_type_ids = torch.stack(input_relation_type_ids)
        
        inputs = self.processor.feature_extractor(raw_speech = waveforms,
                                                  sampling_rate = 16000,
                                                  padding = 'max_length',
                                                  return_tensors = 'pt',
                                                  return_attention_mask = True
                                                  )
        sentence_results = self.processor.tokenizer(sentences,
                                                padding=True,
                                                truncation = True,
                                                max_length =448 ,
                                                return_tensors="pt")
        
        target_results = self.processor.tokenizer(target,
                                                padding=True,
                                                truncation = True,
                                                max_length =448 ,
                                                return_tensors="pt")
        decoder_input_result = self.processor.tokenizer(decoder_input_texts,
                                                padding=True,
                                                truncation = True,
                                                max_length =448 ,
                                                return_tensors="pt")
        stage = torch.tensor([1]) if stage=="train" else torch.tensor([0])
        return inputs,sentence_results,target_results,decoder_input_result,input_relation_type_ids,stage

    def collate_infer(self,data):
        collate_infer_data = {}
        for data_item in data:
            for k,v in data_item.items():
                if k in collate_infer_data:
                    collate_infer_data[k].append(v)
                else:
                    collate_infer_data[k] = []
        collate_infer_data_final = {}
        for k,v in collate_infer_data.items():
            collate_infer_data_final[k] = self.collate_train(v)
        return collate_infer_data_final

if __name__ == "__main__":
    class Hypernum:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    hypernum = Hypernum(
        data_path = r"Datasets/speech_ReTracred",
        audio_file = r"/root/autodl-tmp/speech_ReTracred_audio/speech_ReTracred_audio",
        whisper_model_dir = r"PretrainedSpeechModel/whisper-small.en",
        data_type = "train",
        train_order_views = [0,1,2,3,4,5],
        predict_order_views = [0,1,2,3,4,5],
        del_no_relation_type = False,
        batch_size = 16,
        epoch_num = 50,
        warmup_rate = 0.1,
        learning_rate =2e-5,
        seed = 42,
        num_beams = 3,
        decoder_max_length = 448,
        precision = '16-mixed',
        gpu_device = [0],
        dataloader_numworkers = 4,
        skip_val_epochs =10,
    )
    import tqdm
    data_module = SpeechReDatamodule(hypernum)#data_module必须在model前初始化，设计到tokenizer的传递与改变
    orders_list = ["hrt","htr","rht","rth","thr","trh"]
    for order in orders_list:
        dataset = SpeechRE_Dataset(
                        hypernum.data_path,
                        hypernum.audio_file,
                        "train",
                        order,
                        hypernum.del_no_relation_type,
                        hypernum
                                                )
        pred_num = 0
        pred_true_num=0
        gold_num = 0
        for data_item in tqdm.tqdm(dataset):
            triple_label_sam_list = []
            genetare_label = data_item[2]
            entity_list = data_item[4]
            triple_list = data_item[5]
            for t in triple_list:
                head_e = entity_list[t[0]]
                tail_e = entity_list[t[1]]
                type_r = t[2] if t[2].startswith("<") else "<{}>".format(t[2])
                triple_label_sam_list.append([head_e[3],head_e[4],type_r,tail_e[3],tail_e[4]])
            

            triple_list_test = dataset.convert_tree_squeeze_to_triplet(order,genetare_label)

            gold_num += len(triple_label_sam_list)
            pred_num += len(triple_list_test)

            for item in triple_list_test:
                if item in triple_label_sam_list:
                    pred_true_num+=1
            P = pred_true_num/pred_num
            R = pred_true_num/gold_num
            F1 = 2/(1/(P+0.00000000001)+1/(R+0.00000000001))
            print("顺序{},P{},R{},F1值{}".format(order,P,R,F1))
            
        
        
                                             


        