from minigpt4.datasets.datasets.cal_vqa_datasets import CALVQADataset
from torch.utils.data import DataLoader, DistributedSampler
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torchvision import transforms
import torch


backbone = torch.load("/root/autodl-tmp/VLR_project/ckpt/minigptv2_backbone.pth")
ckpt = torch.load("/root/autodl-tmp/VLR_project/ckpt/20241126081/checkpoint_best.pth")

print(backbone["optimizer"].keys())
print(ckpt["optimizer"].keys())




# processor = lambda x: torch.Tensor([1])
# vis_processor = transforms.ToTensor()
# dataset = CALVQADataset(processor, vis_processor, 
#                         "/root/autodl-tmp/VLR_project/data/cal_data/train", 
#                         ["/root/autodl-tmp/VLR_project/data/cal_data/train_data.json"])

# loader = DataLoader(dataset, 1, True)
# loader = PrefetchLoader(loader)
# loader = IterLoader(loader, use_distributed=True)

# for data in loader:
#     print(data)
#     assert False



# demo = \
# {
#     'arch': 'minigpt_v2', 'image_size': 448, 'drop_path_rate': 0, 
#     'use_grad_checkpoint': False, 'vit_precision': 'fp16', 'freeze_vit': True, 
#     'prompt': '', 'llama_model': '/root/autodl-tmp/VLR_project/llama/llama', 
#     'lora_r': 64, 'lora_alpha': 16, 'model_type': 'pretrain', 'max_txt_len': 500, 
#     'end_sym': '</s>', 'low_resource': True, 'prompt_template': '[INST] {} [/INST]', 
#     'ckpt': '/root/autodl-tmp/VLR_project/ckpt/minigptv2_backbone.pth', 'device_8bit': 0
# }

# train = \
# {
#     'arch': 'minigpt_v2', 'image_size': 448, 'drop_path_rate': 0, 
#     'use_grad_checkpoint': True, 'vit_precision': 'fp16', 'freeze_vit': True, 
#     'prompt': '', 'llama_model': '/root/autodl-tmp/VLR_project/llama/llama', 
#     'lora_r': 64, 'lora_alpha': 16, 'model_type': 'pretrain', 'max_txt_len': 1024, 
#     'end_sym': '</s>', 'ckpt': '/root/autodl-tmp/VLR_project/ckpt/minigptv2_backbone.pth', 'chat_template': True
# }