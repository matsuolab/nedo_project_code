import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import PeftConfig
from mergoo.safe_saving import save_pretrained
from mergoo.composers.composer_lora_moe import ComposeLoraMoeExperts

class ComposeLoraMoeExpertsWithLMHEAD(ComposeLoraMoeExperts):
    def _is_layer_suitable_for_router(self, layer_identifier, model_layer):
        model_layer_index = [int(x) for x in model_layer.split(".") if x.isdigit()]
        if not model_layer_index:
            valid_layer_index = False
        else:
            if "lm_head" in model_layer.lower():
                return False
            elif "lora" in model_layer.lower():
                assert len(model_layer_index) == 2
            else:
                assert len(model_layer_index) == 1  # [layer index, adapter index]
            valid_layer_index = self._check_moe_layers(model_layer_index[0])

        if (layer_identifier in model_layer) and valid_layer_index:
            return True
        return False

    def _exclude_head(self, targets):
        router_layers = []
        for module in list(targets):
            if "lm_head" not in module:
                router_layers.append(module)
        return router_layers
    
    def _adapter_config_to_dict(self, config: PeftConfig):
        adapter_config_dict = config.to_dict()
        for k, v in adapter_config_dict.items():
            try:
                json.dumps(v)
            except:
                adapter_config_dict[k] = str(v)
        return adapter_config_dict
    
    def compose(self):
        """
        Compose all the experts into a single unified checkpoint.
        """
        expert_num = len(self.config["experts"])
        model_for_load = self._load_base_model(self.config["base_model"])
        self.model_config = model_for_load.config.to_dict()
        count_total_router_layers = 0

        ## load full adapter
        for ix, expert in enumerate(self.config["experts"]):
            adapter_id = expert["model_id"]
            model_for_load.load_adapter(adapter_id, adapter_name=str(ix))
        print("load_base active_adapters", model_for_load.active_adapters)
        print("-"*200)

        ## for attn or mlp adapter
        new_model = self._load_base_model(self.config["base_model"])
        for idx, expert in enumerate(self.config["experts"]):
            state_dict = model_for_load.get_adapter_state_dict(str(idx))
            new_state_dict = {}
            for k in state_dict.keys():
                if "lm_head" not in k:
                    new_state_dict[k] = state_dict[k]

            adapter_id = expert["model_id"]
            adapter_config = PeftConfig.from_pretrained(adapter_id)
            new_targets = self._exclude_head(adapter_config.target_modules)
            adapter_config.target_modules = new_targets
            adapter_config_dict = self._adapter_config_to_dict(adapter_config)
            
            self.config["adapter_configs"].append(adapter_config_dict)
            # check if all the lora are having same target modules
            if "router_layers" in self.config:
                assert (sorted(self.config["router_layers"]) == 
                    sorted(self._exclude_head(adapter_config.target_modules)))
            else:
                self.config["router_layers"] = self._exclude_head(adapter_config.target_modules)
            
            ## only load att or mlp
            new_model.load_adapter(
                adapter_name=f"expert_{idx}",
                adapter_state_dict=new_state_dict,
                peft_config=adapter_config
            )
        print("add expert", new_model.active_adapters)
        print("-"*200)

        ## headは無視する
        # ## for lm_head
        # lora_a = model_for_load.get_adapter_state_dict("0")["lm_head.lora_A.weight"]
        # lora_b = model_for_load.get_adapter_state_dict("0")["lm_head.lora_B.weight"]
        # lora_a_ave = torch.zeros_like(lora_a)
        # lora_b_ave = torch.zeros_like(lora_b)
        # for ix, expert in enumerate(self.config["experts"]):
        #     lora_a_ave += model_for_load.get_adapter_state_dict(str(ix))["lm_head.lora_A.weight"]
        #     lora_b_ave += model_for_load.get_adapter_state_dict(str(ix))["lm_head.lora_B.weight"]
        # lora_a_ave /= expert_num
        # lora_b_ave /= expert_num

        # state_dict_for_lm_head = model_for_load.get_adapter_state_dict(str(idx))
        # for k in state_dict.keys():
        #     if "lm_head" in k:
        #         state_dict_for_lm_head[k] = state_dict[k]
        # new_model.load_adapter(
        #         # adapter_id,
        #         adapter_name="lm_head_mean",
        #         adapter_state_dict=state_dict_for_lm_head
        # )
        # ## create adapter config
        # adapter_id = self.config["experts"][0]["model_id"]
        # lm_head_adapter_config = PeftConfig.from_pretrained(adapter_id)

        # lm_head_adapter_config.target_modules = set(filter(lambda v: "lm_head" in v, list(lm_head_adapter_config.target_modules)))
        # lm_head_adapter_config_dict = self._adapter_config_to_dict(lm_head_adapter_config)
        # self.config["adapter_configs"].append(lm_head_adapter_config_dict)
        
        # print("add lm_head", new_model.active_adapters)
        # print("-"*200)

        if hasattr(new_model, "_tied_weights_keys"):
            self._tied_weights_keys.extend(new_model._tied_weights_keys)


        count_router_layers = 0
        count_averaged_layers = 0
        for layer_name, param in tqdm(new_model.state_dict().items()):
            if (
                sum(
                    [
                        self._is_layer_suitable_for_router(router_layer, layer_name)
                        for router_layer in self.config["router_layers"]
                    ]
                )
                == 1
            ):
                # Note: Index of adapter in the config are kept as adapter names, while saving.
                # Similar should be case while loading the adapters
                assert layer_name in self.state_dict
                count_total_router_layers += 1
                count_router_layers += 1
            else:
                assert layer_name in self.state_dict
                count_averaged_layers += 1

        print(f"count_averaged_layers : {count_averaged_layers}")
        print(f"count_router_layers : {count_router_layers}")
        print(f"count_total_router_layers : {count_total_router_layers}")
        del model_for_load
        del new_model
        gc.collect()    
