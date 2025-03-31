import torch
from torch import nn
from torch import autograd

from model2 import DeeplySupervizedUnet

PTH_PATH = "models/kaggle//production//best.pth"

with torch.no_grad():
    torch_model = DeeplySupervizedUnet().to("cpu")

    torch_model.load_state_dict(
        torch.load(
            PTH_PATH,
            map_location=torch.device("cpu"),
        )["model_state_dict"],
    )
    torch_model.eval()  # disables dropout and batchnorm

    torch_input_1 = torch.randn(1, 1, 64, 256)
    torch_input_2 = torch.randn(1, 1, 32, 512)
    torch_input_3 = torch.randn(1, 1, 16, 1024)
    onnx_program = torch.onnx.export(
        torch_model,
        (torch_input_1, torch_input_2, torch_input_3),
        "net.onnx",
        input_names=["input1", "input2", "input3"],
        output_names=["output"],
        dynamic_axes={
            "input1": {0: "batch_size"},
            "input2": {0: "batch_size"},
            "input3": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
