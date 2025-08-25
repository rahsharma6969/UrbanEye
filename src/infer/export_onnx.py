import argparse, torch
from src.models.unet import UNet

def main(weights, out):
    model = UNet()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 8, 512, 512)  # adjust channels/size
    torch.onnx.export(model, dummy, out, input_names=["x"], output_names=["y"], opset_version=17,
                      dynamic_axes={"x":{0:"N"}, "y":{0:"N"}})
    print("Wrote", out)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.weights, args.out)
