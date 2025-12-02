import os
import torch
import torchaudio
import onnxruntime as ort
from tqdm import tqdm

def load_tokenizer(model_dir):
    onnx_path = os.path.join(model_dir, "speech_tokenizer_v1.onnx")
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    return sess

def extract_s3(sess, wav_path):
    wav, sr = torchaudio.load(wav_path)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    wav = wav.mean(0).unsqueeze(0)   # [1, T]

    feats = wav.numpy().astype("float32")
    feats_len = torch.tensor([feats.shape[1]], dtype=torch.int32).numpy()

    inputs = {
        sess.get_inputs()[0].name: feats,       # feats
        sess.get_inputs()[1].name: feats_len,   # feats_length
    }

    out = sess.run(None, inputs)
    return torch.tensor(out[0]).squeeze(0)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    model_dir = "/root/autodl-tmp/modelscope_cache/iic/CosyVoice-300M"
    sess = load_tokenizer(model_dir)

    wavs = []
    for root, dirs, files in os.walk(args.data_root):
        for f in files:
            if f.endswith(".wav"):
                wavs.append(os.path.join(root, f))

    print(f"Found {len(wavs)} wav files.")
    s3_dict = {}
    failed = 0

    for wav in tqdm(wavs, desc="Extracting S3"):
        try:
            s3 = extract_s3(sess, wav)
            s3_dict[wav] = s3
        except Exception as e:
            failed += 1
            s3_dict[wav] = None

    torch.save(s3_dict, args.save_path)
    print(f"S3 tokens saved to {args.save_path} with {len(s3_dict)} items, failed={failed}")

if __name__ == "__main__":
    main()
