IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


model_name = "internlmv2"
text = '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大 模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\n<image>\nProvide a one-sentence caption for the provided image.<|im_end|><|im_start|>assistant\nplate holding a large donut and three smaller ones.<|im_end|>'
