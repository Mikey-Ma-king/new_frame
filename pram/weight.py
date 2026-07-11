import torch
import collections
import re

def print_state_dict(state_dict, max_items=None):
    """
    打印模型状态字典的内容，可控制打印数量
    Args:
        state_dict: 模型的状态字典
        max_items: 最大打印项目数,默认None打印全部
    """
    if isinstance(state_dict, collections.OrderedDict) or isinstance(state_dict, dict):
        # 使用自然排序来正确反映模型的实际结构顺序
        # items = sorted(state_dict.items())，不能这么排序，否则会出现顺序对不上的情况，比如10排在1和2之间
        items = state_dict.items()
        # 如果max_items为None，则打印所有项
        items_to_print = items if max_items is None else items[:max_items]
        for k, v in items_to_print:
            if isinstance(v, collections.OrderedDict) or isinstance(v, dict) or isinstance(v, list):
                print(f"{k}:")
            elif isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {v}")
    else:
        print(state_dict)

# 加载整个模型的权重
weight_path = r"D:\ultralytics-main\weights\yolo11n.pt"
weights = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=False)
# 得到的是字典，包含模型各种信息的字典

# 打印原始权重字典
print("Original weights dictionary:")
#print_state_dict(weights)

weights_model = weights["model"] if isinstance(weights, dict) else weights
# sequential类型可以直接用索引直接访问层；保持模型参数的正确注册，确保所有参数都能被训练
full_model_state_dict = torch.nn.Module.state_dict(weights_model)
# 得到的是字典，仅包含model权重的字典

# 打印 full_model_state_dict
print("\nFull model state dictionary:")
# print_state_dict(full_model_state_dict)

# prefixes_special = 'model.18.'
# state_dict_special = {k: v for k, v in full_model_state_dict.items() if k.startswith(prefixes_special)}
# print_state_dict(state_dict_special)
# 这部分没有输出，说明字典里就没有18层和其余层，原因？
# 可能是n类型的模型不需要全部的层，这个原因！！

prefixes_PAnet = [f'model.{i}.' for i in range(0, 23)]
# 筛选出PAnet部分的权重
PAnet_state_dict = {k: v for k, v in full_model_state_dict.items() if any(k.startswith(prefix) for prefix in prefixes_PAnet)}
#得到的是仅包含PAnet部分权重的字典
# 打印 PAnet_state_dict
print(f"\nPAnet state dictionary (layers 0 to {22}):")
#print_state_dict(PAnet_state_dict)

# 保存筛选后的权重到新的.pth文件
output_path = r"D:\ultralytics-main\weights\yolo11n-PAnet.pt"
torch.save(PAnet_state_dict, output_path)

prefixes = [f'model.{i}.' for i in range(0, 11)]
# 筛选出backbone部分的权重
backbone_state_dict = {k: v for k, v in full_model_state_dict.items() if any(k.startswith(prefix) for prefix in prefixes)}
#得到的是仅包含backbone部分权重的字典
# 打印 backbone_state_dict
print(f"\nBackbone state dictionary (layers 0 to {10}):")
#print_state_dict(backbone_state_dict)

# 保存筛选后的权重到新的.pth文件
output_path = r"D:\ultralytics-main\weights\yolo11n-backbone.pt"
torch.save(backbone_state_dict, output_path)

prefixes_Full = [f'model.{i}.' for i in range(0, 24)]
# 筛选出Full部分的权重
Full_state_dict = {k: v for k, v in full_model_state_dict.items() if any(k.startswith(prefix) for prefix in prefixes_Full)}
#得到的是仅包含Full部分权重的字典
# 打印 Full_state_dict
print(f"\nFull state dictionary (layers 0 to {23}):")
#print_state_dict(Full_state_dict)

# 保存筛选后的权重到新的.pth文件
output_path = r"D:\ultralytics-main\weights\yolo11n-Full.pt"
torch.save(Full_state_dict, output_path)