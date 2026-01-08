import torch
import timm
import detectors
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
from pathlib import Path
import torchvision.transforms as T
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
import onnxruntime as ort
import numpy as np
from torchvision.utils import save_image

def test_models():
    model_name = "resnet18_svhn"
    folder = Path("models")
    fp32_path = str(folder / f"{model_name}.onnx")
    int8_path = str(folder / f"{model_name}_8bit.onnx")

    # Load a sample from SVHN test set
    transform = T.Compose([
        T.Resize((32, 32)),  # Resize to model's expected input
        T.ToTensor(),
    ])
    svhn_test = SVHN(root="data", split="test", download=True, transform=transform)
    sample_loader = DataLoader(svhn_test, batch_size=1, shuffle=True)

    model = timm.create_model(model_name, pretrained=True).eval()
    ort_sess = ort.InferenceSession(fp32_path)
    ort_sess_int8 = ort.InferenceSession(int8_path)

    for _ in range(10):
        sample_img, label = next(iter(sample_loader))  # Get one sample

        save_image(sample_img[0], "sample.png")
        print("\n-----------------------------------")
        print(sample_img.shape)
        # print min and max values of the tensor
        print("Min pixel value:", torch.min(sample_img).item())
        print("Max pixel value:", torch.max(sample_img).item())
        print("Testing with SVHN sample, label =", label.item())

        # 1. Test PyTorch model
        with torch.no_grad():
            torch_out = model(sample_img)
        print("PyTorch output shape:", torch_out.shape)

        # 2. Test FP32 ONNX model
        onnx_out = ort_sess.run(
            None,
            {"input": sample_img.numpy().astype(np.float32)}
        )[0]
        print("ONNX FP32 output shape:", onnx_out.shape)

        # 3. Test Int8 ONNX model
        int8_out = ort_sess_int8.run(
            None,
            {"input": sample_img.numpy().astype(np.float32)}
        )[0]
        print("ONNX Int8 output shape:", int8_out.shape)

        # Optional: print predicted labels
        torch_pred = torch.argmax(torch_out, dim=1).item()
        onnx_pred = np.argmax(onnx_out, axis=1)[0]
        int8_pred = np.argmax(int8_out, axis=1)[0]
        print("Predictions - PyTorch:", torch_pred, "ONNX FP32:", onnx_pred, "ONNX Int8:", int8_pred)



def convert_to_onnx_8bit():
    model_name = "resnet18_svhn"
    print(f"Loading {model_name}...")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    # Get config
    config = model.default_cfg
    input_size = config['input_size']
    
    # Paths
    folder = Path("models")
    fp32_path = str(folder / f"{model_name}.onnx")
    fp32_pre_path = str(folder / f"{model_name}_preprocessed.onnx")
    int8_path = str(folder / f"{model_name}_8bit.onnx")

    # 1. Export to ONNX (FP32)
    dummy_input = torch.randn(1, *input_size)
    print("Exporting FP32 model...")
    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=18,
        do_constant_folding=True
    )

    # 2. Pre-process the model (Fixes shape inference errors)
    print("Pre-processing model to fix shape metadata...")
    quant_pre_process(
        input_model_path=fp32_path,
        output_model_path=fp32_pre_path
    )

    # 3. Apply 8-bit Quantization on the pre-processed model
    print("Quantizing to Int8...")
    quantize_dynamic(
        model_input=fp32_pre_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8
    )

    print(f"Done. Saved to {int8_path}")

if __name__ == "__main__":
    # convert_to_onnx_8bit()
    test_models()