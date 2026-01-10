import torch
import timm
import detectors
from onnxruntime.quantization import quantize_static, QuantType, quant_pre_process, CalibrationDataReader
from pathlib import Path
import torchvision.transforms as T
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
import onnxruntime as ort
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import Subset

batch_size = 8
model_name = "resnet18_svhn"
folder = Path("models")
folder.mkdir(exist_ok=True)
fp32_path = str(folder / f"{model_name}.onnx")
int8_path = str(folder / f"{model_name}_int8.onnx")

# --- 1. Calibration Data Reader ---
class SVHNDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.enum_data = None
        self.dataloader = dataloader
        self.input_name = input_name

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: x.numpy().astype(np.float32)} for x, _ in self.dataloader]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

def convert_to_onnx():
    model = timm.create_model(model_name, pretrained=True).eval()
    input_size = model.default_cfg['input_size']
    fp32_pre_path = str(folder / f"{model_name}_preprocessed.onnx")

    # Export FP32
    dummy_input = torch.randn(batch_size, *input_size)
    torch.onnx.export(model, dummy_input, fp32_path, input_names=['input'], 
                      output_names=['output'], opset_version=18)

    # Pre-process
    quant_pre_process(input_model_path=fp32_path, output_model_path=fp32_pre_path)

    # Setup Calibration Data (using a subset of SVHN)
    transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    dataset = SVHN(root="data", split="test", download=True, transform=transform)
    # Use ~100-200 samples for calibration
    indices = list(range(200))
    dataset = Subset(dataset, indices)
    calib_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dr = SVHNDataReader(calib_loader, "input")

    # Static Quantization
    quantize_static(
        model_input=fp32_pre_path,
        model_output=int8_path,
        calibration_data_reader=dr,
        quant_format=ort.quantization.QuantFormat.QDQ, # Recommended for newer hardware
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8
    )
    print(f"Static Quantization complete: {int8_path}")

def test_models():
    # Load a sample from SVHN test set
    transform = T.Compose([
        T.Resize((32, 32)),  # Resize to model's expected input
        T.ToTensor(),
    ])
    svhn_test = SVHN(root="data", split="test", download=True, transform=transform)
    sample_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=True)

    model = timm.create_model(model_name, pretrained=True).eval()
    ort_sess = ort.InferenceSession(fp32_path)
    ort_sess_int8 = ort.InferenceSession(int8_path)

    for _ in range(3):
        sample_img, label = next(iter(sample_loader))  # Get one batch
        batch_n = sample_img.size(0)
            
        print("\n-----------------------------------")
        print(sample_img.shape)
        print("Min pixel value:", torch.min(sample_img).item())
        print("Max pixel value:", torch.max(sample_img).item())

        # 1. Test PyTorch model (batch)
        with torch.no_grad():
            torch_out = model(sample_img)
        print("PyTorch output shape:", torch_out.shape)

        # 2. Test FP32 ONNX model (batch)
        onnx_out = ort_sess.run(
            None,
            {"input": sample_img.numpy().astype(np.float32)}
        )[0]
        print("ONNX FP32 output shape:", onnx_out.shape)

        # 3. Test Int8 ONNX model (batch)
        int8_out = ort_sess_int8.run(
            None,
            {"input": sample_img.numpy().astype(np.float32)}
        )[0]
        print("ONNX Int8 output shape:", int8_out.shape)

        # Per-image predictions
        torch_preds = torch.argmax(torch_out, dim=1).cpu().tolist()
        onnx_preds = np.argmax(onnx_out, axis=1).tolist()
        int8_preds = np.argmax(int8_out, axis=1).tolist()
        for i in range(batch_n):
            save_image(sample_img[i], f"sample_{i}.png")
            print(f"Image {i}: label={int(label[i].item())}  preds - PyTorch:{torch_preds[i]}  ONNX FP32:{onnx_preds[i]}  ONNX Int8:{int8_preds[i]}")


if __name__ == "__main__":
    convert_to_onnx()
    test_models()