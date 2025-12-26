import sys
import os
import argparse
import torch
import torchaudio

# Get absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
cosyvoice_root = os.path.join(project_root, 'CosyVoice')

# Add CosyVoice paths
sys.path.append(cosyvoice_root)
sys.path.append(os.path.join(cosyvoice_root, 'third_party/Matcha-TTS'))

try:
    from cosyvoice.cli.cosyvoice import AutoModel
except ImportError as e:
    print(f"Error importing CosyVoice: {e}")
    print("Please make sure you are running this script in the 'cosyvoice' conda environment.")
    sys.exit(1)

def generate(text, output_path, model_dir, prompt_audio=None, prompt_text=None):
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(cosyvoice_root, model_dir)
        
    print(f"Loading model from {model_dir}...")
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        sys.exit(1)

    cosyvoice = AutoModel(model_dir=model_dir)
    
    print(f"Generating audio for text: {text}")
    
    if prompt_audio and os.path.exists(prompt_audio):
        print(f"Using Zero-shot Voice Cloning with prompt audio: {prompt_audio}")
        if not prompt_text:
            print("Warning: Prompt text not provided for zero-shot inference. Using '你好' as placeholder, quality might be affected.")
            prompt_text = "你好"
            
        # CosyVoice's inference_zero_shot expects the prompt_speech_16k to be a Tensor if it's loaded, 
        # BUT looking at the error trace:
        # File "/.../cosyvoice/cli/frontend.py", line 118, in _extract_speech_feat
        # speech = load_wav(prompt_wav, 24000)
        # It seems CosyVoice's frontend tries to load the wav AGAIN even if we pass a tensor?
        # Let's check the signature of inference_zero_shot in CosyVoice.
        # If we pass a Tensor, it might be treating it as a file path string in some internal call.
        
        # Actually, looking at the error: TypeError: Invalid file: tensor(...)
        # It confirms that 'prompt_speech_16k' is being passed to something that expects a filename string.
        
        # Let's try passing the PATH directly if the API supports it, OR check if we are using the API correctly.
        # The error happens inside `self.frontend.frontend_zero_shot(..., prompt_wav, ...)`
        # And then `_extract_speech_feat(prompt_wav)` calls `load_wav(prompt_wav, 24000)`.
        # This strongly suggests that `inference_zero_shot` expects a FILE PATH string for the prompt audio, not a Tensor.
        
        # So we should NOT load it manually here.
        output = cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio, stream=False)
    else:
        # Use SFT model with a default speaker
        # Note: '中文女' is a default speaker in SFT model.
        print("Using SFT inference (Default Speaker)")
        output = cosyvoice.inference_sft(text, '中文女', stream=False)
    
    all_audio_segments = []
    for i, j in enumerate(output):
        all_audio_segments.append(j['tts_speech'])
    
    if all_audio_segments:
        # Concatenate all audio segments along the time dimension (dim=1)
        final_audio = torch.cat(all_audio_segments, dim=1)
        print(f"Generated {len(all_audio_segments)} audio segments.")
        torchaudio.save(output_path, final_audio, cosyvoice.sample_rate)
        print(f"Audio saved to {output_path}")
    else:
        print("No audio generated.")

def load_wav(wav_path, target_sr):
    speech, sample_rate = torchaudio.load(wav_path)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        speech = transform(speech)
    return speech

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="pretrained_models/CosyVoice-300M-SFT")
    parser.add_argument("--prompt_audio", type=str, default=None, help="Path to prompt audio for zero-shot cloning")
    parser.add_argument("--prompt_text", type=str, default=None, help="Text content of the prompt audio")
    args = parser.parse_args()
    
    generate(args.text, args.output, args.model_dir, args.prompt_audio, args.prompt_text)
