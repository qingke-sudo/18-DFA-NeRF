import sys
import os
import argparse
import subprocess
import shutil

# Get absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
dfa_nerf_root = os.path.join(project_root, 'DFA-NeRF')

def generate(audio_path, output_video_path):
    print(f"Processing audio: {audio_path}")
    
    # 1. Audio Feature Extraction
    wav2exp_dir = os.path.join(dfa_nerf_root, 'data_util/wav2exp')
    audio_feature_file = audio_path.replace('.wav', '.pt')
    
    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)

    cmd_audio = [
        sys.executable, "test_w2l_audio.py",
        "--input_path", audio_path,
        "--save_path", audio_path.replace('.wav', '.pt') 
    ]
    
    print(f"Running audio feature extraction in {wav2exp_dir}...")
    try:
        subprocess.run(cmd_audio, cwd=wav2exp_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Audio feature extraction failed: {e}")
        print("Ensure you have the necessary weights in data_util/wav2exp/models/")
        sys.exit(1)
    
    if not os.path.exists(audio_feature_file):
        print(f"Error: Expected output file {audio_feature_file} was not created.")
        sys.exit(1)

    # 2. NeRF Generation
    print("Starting NeRF generation...")
    
    # Prepare paths
    dataset_dir = os.path.join(dfa_nerf_root, 'dataset/obama')
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} not found. Please setup the Obama dataset.")
        sys.exit(1)
        
    target_aud_file = os.path.join(dataset_dir, 'temp_aud.pt')
    shutil.copy(audio_feature_file, target_aud_file)
    
    nerf_script = os.path.join(dfa_nerf_root, 'NeRFs/DFANeRF/run_nerf_com_trainExpLater.py')
    
    # Arguments based on test_obama.sh
    cmd_nerf = [
        sys.executable, nerf_script,
        "--config", "dataset/obama/HeadNeRF_config_ba.txt",
        "--aud_file", "temp_aud.pt",
        "--datadir", "dataset/obama",
        "--expname", "obama_TrainExpLater_smoMix",
        "--render_person",
        "--render_video",
        "--test_file", "transforms_train_ba.json",
        "--noexp_iters", "400000",
        "--resume", "dataset/train_together/obama_TrainExpLater_smoMix/240000.tar",
        "--last_dist", "1e10",
        "--concate_bg",
        "--N_rand", "2048",
        "--sample_rate", "0",
        "--i_print", "100",
        "--i_test_person", "10000",
        "--chunk", "2048",
        "--win_size", "16",
        "--smo_size", "4",
        "--smo_torse_size", "8",
        "--train_together",
        "--i_weights", "100000",
        "--all_speaker",
        "--sample_rate_mouth", "0",
        "--lrate_decay", "500",
        "--lrate", "5e-4",
        "--use_et_embed",
        "--nosmo_iters", "300000",
        "--dim_signal", "96",
        "--dim_aud", "96",
        "--n_object", "1",
        "--N_iters", "600000",
        "--use_deformation_field",
        "--exp_file", "obama_64_32.pt",
        "--use_ba"
    ]
    
    try:
        subprocess.run(cmd_nerf, cwd=dfa_nerf_root, check=True)
        
        # Find the output video
        # Output is in dataset/train_together/obama_TrainExpLater_smoMix/obama/person/render_com/
        results_dir = os.path.join(dfa_nerf_root, 'dataset/train_together/obama_TrainExpLater_smoMix/obama/person/render_com')
        
        # Find the latest mp4 file in results
        list_of_files = []
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith(".mp4"):
                    list_of_files.append(os.path.join(root, file))
        
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            shutil.copy(latest_file, output_video_path)
            print(f"Video generated and saved to {output_video_path}")
        else:
            print("NeRF ran but no video output found in results directory.")
            
    except subprocess.CalledProcessError as e:
        print(f"NeRF generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    generate(args.audio, args.output)
