from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('misdelivery/Mujina-test')
tokenizer = AutoTokenizer.from_pretrained('misdelivery/Mujina-test')
torch.save(model, "mujina_test_pth/mujina_test.pth")
tokenizer.save_pretrained('mujina_test_pth')

from torchtune.utils import FullModelHFCheckpointer
from torchtune.models import convert_weights
import torch

# チェックポイントファイルのリストを生成
checkpoint_files = [f'model-{i:05d}-of-00023.safetensors' for i in range(1, 24)]

# FullModelHFCheckpointerのインスタンスを作成
checkpointer = FullModelHFCheckpointer(
    checkpoint_dir='/home/ext_kan_hatakeyama_s_gmail_com/.cache/huggingface/hub/models--misdelivery--Mujina-test/snapshots/9f95ff89242f7924c3ed65666437001750f66168',
    checkpoint_files=checkpoint_files,
    output_dir='/storage5/llm/codes/2_pretrain/mujina_test_pth',
    model_type='LLAMA3'
)

print("loading checkpoint")
sd = checkpointer.load_checkpoint()
sd = convert_weights.tune_to_meta(sd['model'])

print("saving checkpoint")
torch.save(sd, "/storage5/llm/codes/2_pretrain/mujina_test_pth/checkpoint.pth")