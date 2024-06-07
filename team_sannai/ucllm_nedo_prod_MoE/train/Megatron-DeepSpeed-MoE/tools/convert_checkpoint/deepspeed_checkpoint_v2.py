import os
from typing import Dict
import torch 

from deepspeed.checkpoint.reshape_3d_utils import model_3d_desc
from deepspeed.checkpoint.reshape_utils import (basic_folder_validation, merge_state, partition_data, get_files, get_files_with_prefix)

from deepspeed.checkpoint.constants import (MODEL_FILE_PREFIX, LAYER_FILE_PREFIX)

from deepspeed.checkpoint.reshape_meg_2d import reshape_meg_2d_parallel, meg_2d_parallel_map
from deepspeed.checkpoint.zero_checkpoint import ZeROCheckpoint
from deepspeed.checkpoint.constants import *

ZERO_FILE_PREFIX = 'zero_pp_rank_'
LAYER_FILE_PREFIX = 'layer_'
MP_RANK_FILE_PREFIX = 'mp_rank_'
EMBEDDING_LAYER_INDEX = 0
FINAL_LAYER_NORM_INDEX = -2
LM_HEAD_INDEX = -1
ARGS_KEY = 'args'
ITERATION_KEY = 'iteration'
CHECKPOINT_INFO_KEY = 'checkpoint_info'
SEQUENTIAL_LAYERS = [
    'input_layernorm.weight', 'input_layernorm.bias',
    'self_attention.dense.bias',
    'post_attention_layernorm.weight', 'post_attention_layernorm.bias',
    'mlp.dense_4h_to_h.bias',
    'position_embeddings.weight'
]

LAYER_CONCAT_DIM = {
    'self_attention.dense.weight': 1,
    'mlp.dense_4h_to_h.weight': 1
}

class DeepSpeedCheckpoint(object):
    def __init__(self, dir, tp_degree=None, pp_degree=None, dp_degree=None, no_pp=False):
        self.dir = dir
        pipeline_parallel=len(get_files_with_prefix(get_files(dir), LAYER_FILE_PREFIX)) > 0 #
        #self._validate_folder(dir)
        self._validate_folder(dir, pipeline_parallel)
        self.no_pp= 0 if pipeline_parallel==True else 1 

        self.zero_checkpoint = ZeROCheckpoint(dir)

        self.file_list = get_files(dir)
        self.layer_files = get_files_with_prefix(self.file_list, LAYER_FILE_PREFIX)
        self.mp_rank_files = get_files_with_prefix(self.file_list, MODEL_FILE_PREFIX)

        self.layer_keys = self._get_layer_keys()
        self.layer_count = len(self.layer_keys)

        self.tp_degree = self.zero_checkpoint.get_src_tp_degree() if tp_degree is None else tp_degree
        self.pp_degree = self.zero_checkpoint.get_src_pp_degree() if pp_degree is None else pp_degree
        #self.pp_degree = self.pp_degree if self.pp_degree > 0 else 1
        self.dp_degree = self.zero_checkpoint.get_src_dp_degree() if dp_degree is None else dp_degree


        self.original_world_size = self.zero_checkpoint.get_src_tp_degree() * self.zero_checkpoint.get_src_pp_degree(
        ) * self.zero_checkpoint.get_src_dp_degree()
        self.world_size = self.tp_degree * self.pp_degree * self.dp_degree

        self.old_2d_map = meg_2d_parallel_map(self.zero_checkpoint.get_src_pp_degree(),
                                              self.zero_checkpoint.get_src_tp_degree())
        self.old_2d_map.simple_init()
        self.new_2d_map = reshape_meg_2d_parallel(old_pp_degree=self.zero_checkpoint.get_src_pp_degree(),
                                                  old_tp_degree=self.zero_checkpoint.get_src_tp_degree(),
                                                  new_pp_degree=self.pp_degree,
                                                  new_tp_degree=self.tp_degree)

        if self.is_change_pp_degree() or self.is_change_tp_degree() or self.is_change_dp_degree():
            self.zero_checkpoint.reshape(model_3d_desc(self.pp_degree, self.tp_degree, self.dp_degree))

        self.global_state = {}
    
        self._sanity_check()
        self.pp_to_transformer_map = self._build_pp_transformer_map()
        self.transformer_file_map = self._build_transformer_file_map()
        #if not self.no_pp:
        self.tp_to_embedding_map = self._build_tp_other_layer_map(EMBEDDING_LAYER_INDEX)
        self.tp_to_final_norm_map = self._build_tp_other_layer_map(FINAL_LAYER_NORM_INDEX)
        self.tp_to_lm_head_map = self._build_tp_other_layer_map(LM_HEAD_INDEX)
        self._build_global_state()
    
    def is_change_tp_degree(self):
        return self.tp_degree != self.zero_checkpoint.get_src_tp_degree()

    def is_change_pp_degree(self):
        return self.pp_degree != self.zero_checkpoint.get_src_pp_degree()

    def is_change_dp_degree(self):
        return self.dp_degree != self.zero_checkpoint.get_src_dp_degree()

    def show_2d_mapping(self):
        print(f'reshaped 2d map ---- begin')

        for i in range(self.pp_degree):
            for j in range(self.tp_degree):
                file_list = self.get_2d_parallel_files(pp_index=i, tp_index=j)
                print(f'[{i}, {j}] = {file_list}')

        print(f'reshaped 2d map ---- end')

    def show_tp_embedding_map(self):
        self._dump_mapping(self.tp_to_embedding_map, 'tp_to_embedding_layers')

    def show_tp_final_norm_map(self):
        self._dump_mapping(self.tp_to_final_norm_map, 'tp_to_final_norm_layers')

    def show_pp_tranformer_map(self):
        self._dump_mapping(self.pp_to_transformer_map, 'pp_to_tranformer_layers')

    def show_transformer_file_map(self):
        self._dump_mapping(self.transformer_file_map, 'rank_to_tranformer_files')

    def _build_global_state(self):
        sd = torch.load(self.mp_rank_files[0], map_location=torch.device('cpu'))
        self.global_state[ITERATION_KEY] = sd.get(ITERATION_KEY, 0)
        self.global_state[ARGS_KEY] = sd.get(ARGS_KEY, None)

    def get_zero_checkpoint_state(self, pp_index, tp_index, dp_index) -> dict:
        return self.zero_checkpoint.get_state_for_rank(pp_index=pp_index,
                                                       tp_index=tp_index,
                                                       dp_index=dp_index,
                                                       keys_to_ignore=[PARAM_SHAPES])

    def get_zero_files(self, pp_index, tp_index, dp_index) -> list:
        return self.zero_checkpoint.get_files_for_rank(pp_index=pp_index, tp_index=tp_index, dp_index=dp_index)

    def get_embedding_layer_id(self):
        return self.layer_keys[EMBEDDING_LAYER_INDEX]

    def get_lm_head_id(self):
        return self.layer_keys[LM_HEAD_INDEX]

    def get_final_norm_layer_id(self):
        return self.layer_keys[FINAL_LAYER_NORM_INDEX]

    def get_iteration(self):
        if not ITERATION_KEY in self.global_state:
            sd = torch.load(self.mp_rank_files[0], map_location=torch.device('cpu'))
            self.global_state[ITERATION_KEY] = sd.get(ITERATION_KEY, 0)

        return self.global_state[ITERATION_KEY]

    def get_embedding_state(self, tp_index: int) -> Dict:
        assert tp_index in self.tp_to_embedding_map.keys()
        sd_list = [torch.load(fname, map_location=torch.device('cpu')) for fname in self.tp_to_embedding_map[tp_index]]
        sd = self._merge_state_dicts(sd_list)
        return sd

    def get_embedding_files(self, tp_index: int) -> list:
        assert tp_index in self.tp_to_embedding_map.keys()
        return self.tp_to_embedding_map[tp_index]

    def get_lm_head_state(self, tp_index: int) -> Dict:
        assert tp_index in self.tp_to_embedding_map.keys()
        sd_list = [torch.load(fname, map_location=torch.device('cpu')) for fname in self.tp_to_lm_head_map[tp_index]]
        sd = self._merge_state_dicts(sd_list)
        return sd

    def get_lm_head_files(self, tp_index: int) -> list:
        assert tp_index in self.tp_to_lm_head_map.keys()
        return self.tp_to_lm_head_map[tp_index]
    
    def _get_checkpoint_value(self, key):
        if not key in self.global_state:
            sd = torch.load(self.mp_rank_files[0], map_location=torch.device('cpu'))
            self.global_state[key] = sd.get(key, None)
        #print(f"self.global_state {self.global_state}")

        return self.global_state[key]

    def get_args(self):
        if not ARGS_KEY in self.global_state:
            sd = torch.load(self.mp_rank_files[0], map_location=torch.device('cpu'))
            self.global_state[ARGS_KEY] = sd.get(ARGS_KEY, None)

        return self.global_state[ARGS_KEY]

    def get_checkpoint_info(self, info_key=CHECKPOINT_INFO_KEY):
        return self._get_checkpoint_value(info_key)   

    def get_2d_parallel_state(self, tp_index: int, pp_index: int) -> dict:
        assert tp_index < self.tp_degree
        assert pp_index < self.pp_degree
        fname_list = self.get_2d_parallel_files(tp_index=tp_index, pp_index=pp_index)
        sd_list = [torch.load(fname, map_location=torch.device('cpu')) for fname in fname_list]

        merged_sd = None
        for sd in sd_list:
            if merged_sd is None:
                merged_sd = sd
            else:
                merged_sd = merge_state(merged_sd, sd)

        return merged_sd

    def get_transformer_state(self, tp_index: int, pp_index: int) -> list:
        assert tp_index < self.tp_degree
        assert pp_index < self.pp_degree
        t_list = []
        for fname_list in self.transformer_file_map[(tp_index, pp_index)]:
            sd_list = [torch.load(fname, map_location=torch.device('cpu')) for fname in fname_list]
            sd = self._merge_state_dicts(sd_list)
            t_list.append(sd)
        return t_list       
        
    def get_pp_transformer_map(self, pp_index: int) -> list:
        assert pp_index < self.pp_degree
        return self.pp_to_transformer_map[pp_index]

    def get_final_norm_state(self, tp_index: int) -> Dict:
        assert tp_index in self.tp_to_final_norm_map.keys()
        sd = torch.load(self.tp_to_final_norm_map[tp_index][0], map_location=torch.device('cpu'))
        return sd

    def get_final_norm_files(self, tp_index: int) -> list:
        assert tp_index in self.tp_to_final_norm_map.keys()
        return self.tp_to_final_norm_map[tp_index]

    def _build_tp_other_layer_map(self, layer_index:int):
        data_map = {}#
        if len(self.layer_files)<1:#
            return data_map#
        assert layer_index < len(self.layer_files)
        layer_files = self._get_files_with_prefix(self.layer_files, self.layer_keys[layer_index])
        layer_file_partitions = self._partition_data(layer_files, self.tp_degree)
        data_map = {i:flist for i, flist in enumerate(layer_file_partitions)}
        return data_map

    def get_2d_parallel_files(self, tp_index: int, pp_index: int) -> list:
        assert tp_index < self.tp_degree
        assert pp_index < self.pp_degree
        file_indices = self.new_2d_map.get_data(pp_index=pp_index, tp_index=tp_index)
        return [self.mp_rank_files[i] for i in file_indices]

    def _build_pp_transformer_map(self):
        data_map = {}
        if self.pp_degree>0:
            transformer_layers = self.layer_keys[1:-2]
            layers_per_pp = len(transformer_layers) // self.pp_degree
            data_map = {i:transformer_layers[i*layers_per_pp:(i+1)*layers_per_pp] for i in range(0, self.pp_degree)}
        return data_map

    def _dump_mapping(self, data_map, map_tag = None):
        if map_tag is not None:
            print(f'Dump mapping: {map_tag}')
        for k, v in data_map.items():
            print(f'{k} = {v}')

    def _build_transformer_file_map(self):
        transformer_layer_keys = self.layer_keys[1:-2]
        file_map = {}
        layers_per_pp=1#
        if self.pp_degree>0:
            layers_per_pp=len(transformer_layer_keys) // self.pp_degree
        for key_index, layer_key in enumerate(transformer_layer_keys):
            pp_index = key_index // layers_per_pp
            layer_files = self._get_files_with_prefix(self.layer_files, layer_key)
            layer_file_partitions = self._partition_data(layer_files, self.tp_degree)
            for tp_index in range(self.tp_degree):
                map_key = (tp_index, pp_index)
                if not map_key in file_map.keys():
                    file_map[map_key] = []
                file_map[map_key].append(layer_file_partitions[tp_index])
        
        return file_map
        
    def _sanity_check(self):
        assert len(self.mp_rank_files) % self.tp_degree == 0
        #assert len(self.zero_files) % (self.pp_degree * self.tp_degree) == 0
        if not self.no_pp:
            assert len(self.layer_keys) > 2
            assert (len(self.layer_keys) - 3) % self.pp_degree == 0
     
    def _get_files_with_prefix(self, all_files, prefix):
        file_list = []
        for file_path in all_files:
            _, fname = os.path.split(file_path)
            if fname.startswith(prefix):
                file_list.append(file_path)
        
        return sorted(file_list)

    def validate_files(self):
        for file in self.file_list:
            if not os.path.isfile(file):
                print(f'Error: {file} is not existent')
        
    def _get_files(self, dir):
        file_list = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    def _get_layer_keys(self):
        key_set = set()
        key_len = len(LAYER_FILE_PREFIX) + 2
        for file_path in self.layer_files:
            _, fname = os.path.split(file_path)
            key_set.add(fname[:key_len])
        return sorted(list(key_set))

    def _partition_data(self, data_list, num_partitions):
        num_elems = len(data_list)
        assert num_elems % num_partitions == 0
        partition_size = num_elems // num_partitions
        partitions_list = [data_list[i:i+partition_size] for i in range(0, num_elems, partition_size)]
        return partitions_list

    def _merge_state_dicts(self, sd_list):
        merged_sd = {}
        for key in sd_list[0].keys():
            if not key in SEQUENTIAL_LAYERS:
                cat_dim = LAYER_CONCAT_DIM.get(key, 0)
                merged_sd[key] = torch.cat([sd[key] for sd in sd_list], dim=cat_dim)
            else:
                merged_sd[key] = sd_list[0][key]
        return merged_sd

    def _validate_folder(self, dir, pipeline_parallel): #
        basic_folder_validation(dir)

        file_list = get_files(dir)
        file_prefix_list = [MODEL_FILE_PREFIX] #

        if pipeline_parallel:#
            file_prefix_list.extend([LAYER_FILE_PREFIX, f'{LAYER_FILE_PREFIX}01'])#

        #for file_prefix in [MODEL_FILE_PREFIX, LAYER_FILE_PREFIX, f'{LAYER_FILE_PREFIX}01']:
        for file_prefix in file_prefix_list:
            ckpt_files = get_files_with_prefix(file_list, file_prefix)
            assert len(
                ckpt_files
            ) > 0, f'{dir} seems a bogus DeepSpeed checkpoint folder: Cannot find {file_prefix}* files in there.'
