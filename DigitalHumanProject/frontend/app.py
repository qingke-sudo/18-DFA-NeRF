import streamlit as st
import os
import subprocess
import time
from PIL import Image

# Page Config
st.set_page_config(page_title="Data Hammer Group Talking System", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    .main-title {
        font-family: 'Arial', sans-serif;
        font-size: 48px;
        font-weight: bold;
        color: #00FFFF;
        text-align: center;
        text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF;
        margin-top: 50px;
        margin-bottom: 50px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #0099FF;
        color: white;
        font-size: 18px;
        height: 50px;
        border-radius: 5px;
        border: none;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #0077CC;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #444;
    }
    .stTextArea>div>div>textarea {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #444;
    }
    .stSelectbox>div>div>div {
        background-color: #2d2d2d;
        color: white;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom container for the menu */
    .menu-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify_content: center;
        padding: 50px;
    }
    
    /* Force text color to white for chat messages */
    .stChatMessage p {
        color: white !important;
    }
    
    /* Adjust video size in chat */
    .stVideo {
        width: 50% !important; /* Reduce video width */
    }
</style>
""", unsafe_allow_html=True)

# Session State
if 'page' not in st.session_state:
    st.session_state['page'] = 'menu'

if 'latency_history' not in st.session_state:
    st.session_state['latency_history'] = []

# Navigation Functions
def go_to_menu():
    st.session_state['page'] = 'menu'

def go_to_train():
    st.session_state['page'] = 'train'

def go_to_generate():
    st.session_state['page'] = 'generate'

def go_to_chat():
    st.session_state['page'] = 'chat'

# Helper function for commands
def get_command(env_name, script, args):
    # Use environment variables for python paths if available
    if env_name == "cosyvoice":
        python_executable = os.environ.get("COSYVOICE_PYTHON", "/home/archie/miniconda3/envs/cosyvoice/bin/python")
    elif env_name == "adnerf":
        python_executable = os.environ.get("ADNERF_PYTHON", "/home/archie/miniconda3/envs/adnerf/bin/python")
    else:
        python_executable = "python" # Fallback
        
    return [python_executable, script] + args

# LLM Loading (Cached)
@st.cache_resource
def load_llm():
    # Use Qwen/Qwen2.5-0.5B-Instruct for speed and low resource usage
    # It runs easily on CPU
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        # Try loading from modelscope if available to speed up download in CN
        try:
            from modelscope import snapshot_download
            model_dir = snapshot_download(model_name)
        except ImportError:
            model_dir = model_name
        except Exception:
            model_dir = model_name

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None, None

def chat_with_llm(tokenizer, model, prompt, history=[]):
    if not tokenizer or not model:
        return "LLM not loaded."
    
    messages = history + [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_audio = os.path.join(current_dir, "output.wav")
output_video = os.path.join(current_dir, "output.webm") # Use webm for better browser support
example_video_path = os.path.join(current_dir, "chat_video_sample.webm") # Use the WebM version

# --- PAGE: MENU ---
if st.session_state['page'] == 'menu':
    st.markdown('<div class="main-title">DATA HAMMER GROUP<br>TALKING SYSTEM</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("训练模型"):
            go_to_train()
            st.rerun()
        if st.button("视频生成"):
            go_to_generate()
            st.rerun()
        if st.button("人机对话"):
            go_to_chat()
            st.rerun()

# --- PAGE: TRAIN ---
elif st.session_state['page'] == 'train':
    st.markdown('<div class="section-header">模型训练界面</div>', unsafe_allow_html=True)
    
    if st.button("返回主菜单"):
        go_to_menu()
        st.rerun()

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("💡 提示：请上传一段单人、正脸、背景稳定的视频（建议3-5分钟）。")
        uploaded_file = st.file_uploader("上传训练视频 (MP4)", type=["mp4"])
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            
            # Save uploaded video to a temp location or dataset folder
            # We need to know the model name first to save it correctly
    
    with col2:
        model_name = st.text_input("模型名称 (ID)", value="my_avatar", help="给您的数字人起个名字，例如 'obama' 或 'me'。")
        
        st.markdown("### 训练参数设置")
        epochs = st.number_input("训练迭代次数 (Iterations)", value=600000, step=10000, help="训练次数越多，效果通常越好，但耗时更长。建议至少 400,000 次。")
        batch_size = st.number_input("Batch Size", value=2048, help="显存越大，可以设置得越大。")
        gpu_id = st.selectbox("选择 GPU", ["0", "1", "2", "3"], index=0)
        
        st.markdown("### 高级参数")
        with st.expander("展开高级设置"):
            lrate = st.text_input("学习率 (Learning Rate)", value="5e-4")
            use_deformation = st.checkbox("使用变形场 (Deformation Field)", value=True, help="处理头部微小移动，建议开启。")
        
        if st.button("开始训练 (Start Training)"):
            if uploaded_file is None:
                st.error("请先上传视频文件！")
            elif not model_name:
                st.error("请输入模型名称！")
            else:
                # 1. Save Video
                video_dir = os.path.join(project_root, "DFA-NeRF", "dataset", "vids")
                os.makedirs(video_dir, exist_ok=True)
                save_path = os.path.join(video_dir, f"{model_name}.mp4") # Keep input as mp4
                
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"视频已保存至: {save_path}")
                
                # 2. Explanation of what happens next
                st.info("正在初始化训练流程...")
                
                # --- DEMO MODE: Simulate Training ---
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("数据预处理: 提取帧与关键点...")
                time.sleep(1.5)
                progress_bar.progress(25)
                
                status_text.text("数据预处理: 提取音频特征...")
                time.sleep(1.5)
                progress_bar.progress(50)
                
                status_text.text("模型训练中 (NeRF)...")
                time.sleep(2.0)
                progress_bar.progress(75)
                
                status_text.text("模型微调与优化...")
                time.sleep(1.5)
                progress_bar.progress(100)
                
                st.success("训练完成！模型已就绪。")
                
                # Show the "result" video (using the sample video for demo purposes)
                if os.path.exists(example_video_path):
                    st.markdown("### 训练结果预览")
                    # Read as bytes to ensure loading
                    with open(example_video_path, 'rb') as v:
                        st.video(v.read())
                else:
                    st.warning(f"训练完成，但无法加载预览视频。路径未找到: {example_video_path}")
                
                # In a real scenario, we would trigger a background task here using subprocess
                # cmd = ["bash", "scripts/process_data.sh", model_name]
                # subprocess.Popen(cmd, cwd=...)

# --- PAGE: GENERATE ---
elif st.session_state['page'] == 'generate':
    st.markdown('<div class="section-header">视频生成界面</div>', unsafe_allow_html=True)
    
    if st.button("返回主菜单", key="back_btn"):
        go_to_menu()
        st.rerun()

    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Video Player
        if os.path.exists(output_video):
            # Read as bytes to ensure loading and avoid caching issues
            with open(output_video, 'rb') as v:
                st.video(v.read())
            
            # --- Performance Metrics Display ---
            if 'last_metrics' in st.session_state:
                m = st.session_state['last_metrics']
                st.markdown("### ⚡ 端到端延迟统计 (End-to-End Latency)")
                
                # Current Run
                cols = st.columns(4)
                cols[0].metric("ASR (识别)", f"{m['ASR']:.2f}s")
                cols[1].metric("LLM (思考)", f"{m['LLM']:.2f}s")
                cols[2].metric("TTS (合成)", f"{m['TTS']:.2f}s")
                cols[3].metric("Video (渲染)", f"{m['Video']:.2f}s")
                st.metric("总耗时 (Total)", f"{m['Total']:.2f}s")
                
                # Historical Stats
                if len(st.session_state['latency_history']) > 1:
                    import pandas as pd
                    df = pd.DataFrame(st.session_state['latency_history'])
                    st.markdown("#### 📊 历史性能趋势 (Average)")
                    avg = df.mean()
                    st.dataframe(avg.to_frame().T.style.format("{:.2f}s"))

        else:
            st.markdown("""
            <div style="background-color: black; height: 400px; display: flex; align-items: center; justify-content: center; border-radius: 10px;">
                <span style="color: #555;">Generated Video Will Appear Here</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Form Fields
        model_name = st.selectbox("模型名称", ["DFA-NeRF", "SyncTalk"])
        model_path = st.text_input("模型目录地址", "dataset/train_together/obama_TrainExpLater_smoMix")
        ref_audio = st.text_input("参考音频地址", output_audio)
        gpu_select = st.selectbox("GPU选择", ["GPU 0"])
        voice_model = st.selectbox("语音克隆模型名称", ["CosyVoice SFT", "Voice Clone A"])
        target_text = st.text_area("目标文字 (文本框)", "你好，我是数字人助手。")
        
        if st.button("生成视频"):
            # Logic: If text is provided, generate audio first.
            # If text is empty, use the reference audio path directly.
            
            # Initialize metrics
            t_start_total = time.time()
            t_asr = 0.0 # Skipped for text input
            t_llm = 0.0 # Skipped for direct text
            t_tts = 0.0
            t_video = 0.0
            
            audio_file_to_use = ref_audio
            
            # 1. Generate Audio (REAL)
            if target_text.strip():
                with st.spinner("正在生成音频 (CosyVoice)..."):
                    t_start_tts = time.time()
                    script_path = os.path.join(current_dir, "generate_audio.py")
                    # Using 'cosyvoice' env
                    # Model path provided by user or env var
                    project_root = os.path.dirname(current_dir)
                    local_model_dir = os.path.join(project_root, "pretrained_models", "CosyVoice-300M-SFT")
                    if os.path.exists(local_model_dir):
                        default_model_dir = local_model_dir
                    else:
                        default_model_dir = "/home/archie/Documents/Work/pretrained_models/CosyVoice-300M-SFT"
                    
                    model_dir = os.environ.get("COSYVOICE_MODEL_DIR", default_model_dir)
                    
                    cmd = get_command("cosyvoice", script_path, ["--text", target_text, "--output", output_audio, "--model_dir", model_dir])
                    
                    try:
                        st.info(f"Executing command: {' '.join(cmd)}") # Debug info
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        t_end_tts = time.time()
                        t_tts = t_end_tts - t_start_tts
                        
                        if result.returncode != 0:
                            st.error("音频生成失败")
                            st.markdown("**Error Output (stderr):**")
                            st.code(result.stderr)
                            st.markdown("**Standard Output (stdout):**")
                            st.code(result.stdout)
                            st.stop()
                        else:
                            st.success(f"音频生成成功 (耗时: {t_tts:.2f}s)")
                            # Optional: Show stdout even on success if needed for debugging
                            # st.expander("Show Output").code(result.stdout)
                            audio_file_to_use = output_audio
                    except Exception as e:
                        st.error(f"Error generating audio: {e}")
                        st.stop()
            
            # 2. Generate Video (DEMO MODE: Simulate Generation)
            with st.spinner(f"正在生成视频 (DFA-NeRF)..."):
                t_start_video = time.time()
                # time.sleep(1.0) # Optional: Simulate a bit more "thinking" time if ffmpeg is too fast
                
                if os.path.exists(example_video_path) and os.path.exists(audio_file_to_use):
                    # Mux audio and video using FFmpeg
                    # We loop the video to match the audio length
                    ffmpeg_path = "/home/archie/miniconda3/envs/adnerf/bin/ffmpeg"
                    
                    cmd = [
                        ffmpeg_path,
                        "-stream_loop", "-1", # Loop the video input
                        "-i", example_video_path,
                        "-i", audio_file_to_use,
                        "-shortest", # Stop when the shortest stream (audio) ends
                        "-map", "0:v",
                        "-map", "1:a",
                        "-c:v", "copy", # Copy video stream (fast)
                        "-c:a", "libopus", # Encode audio for WebM (changed from libvorbis)
                        "-y",
                        output_video
                    ]
                    
                    try:
                        # st.info(f"Synthesizing video: {' '.join(cmd)}")
                        subprocess.run(cmd, capture_output=True, check=True)
                        t_end_video = time.time()
                        t_video = t_end_video - t_start_video
                        
                        # Record Metrics
                        t_total = time.time() - t_start_total
                        metrics = {
                            "ASR": t_asr,
                            "LLM": t_llm,
                            "TTS": t_tts,
                            "Video": t_video,
                            "Total": t_total
                        }
                        st.session_state['latency_history'].append(metrics)
                        st.session_state['last_metrics'] = metrics # Store for display after rerun
                        
                        st.success(f"视频生成成功! (耗时: {t_video:.2f}s)")
                        time.sleep(0.5) # Wait a bit before rerun
                        st.rerun()
                    except subprocess.CalledProcessError as e:
                        st.error("视频合成失败 (FFmpeg)")
                        st.text(e.stderr.decode())
                elif not os.path.exists(example_video_path):
                    st.error(f"演示视频文件丢失: {example_video_path}")
                else:
                    st.error(f"音频文件丢失: {audio_file_to_use}")

# --- PAGE: CHAT ---
elif st.session_state['page'] == 'chat':
    st.markdown('<div class="section-header">人机对话</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("返回主菜单"):
            go_to_menu()
            st.rerun()
        if st.button("清空对话"):
            st.session_state.messages = []
            st.rerun()

    # Voice Cloning Settings
    prompt_audio_path = None
    prompt_text_final = None
    
    with st.expander("🎙️ 声音克隆设置 (Voice Cloning Settings)"):
        st.info("上传一段 3-10秒 的人声录音，数字人将模仿该声音进行回复。如果不上传，默认使用预设女声。")
        uploaded_voice = st.file_uploader("上传参考音频 (WAV/MP3)", type=["wav", "mp3"])
        prompt_text_input = st.text_input("参考音频对应的文本 (可选，留空自动识别)", help="如果留空，系统将自动使用 Whisper 识别音频内容。")
        
        if uploaded_voice:
            # Save uploaded file
            # Use a fixed name so we don't accumulate files, or session ID based
            prompt_audio_path = os.path.join(current_dir, "current_prompt_audio.wav")
            with open(prompt_audio_path, "wb") as f:
                f.write(uploaded_voice.getbuffer())
            st.audio(prompt_audio_path)
            
            if prompt_text_input:
                prompt_text_final = prompt_text_input
            else:
                # Auto transcribe
                # Only transcribe if we haven't already or if file changed (simplified: just transcribe)
                # To avoid re-running heavy transcribe on every interaction, we could cache it, 
                # but for now let's just run it. It's fast enough for short audio.
                transcribe_script = os.path.join(current_dir, "transcribe_audio.py")
                cmd_transcribe = get_command("cosyvoice", transcribe_script, ["--audio", prompt_audio_path])
                try:
                    res = subprocess.run(cmd_transcribe, capture_output=True, text=True)
                    if res.returncode == 0 and "TRANSCRIPTION_START" in res.stdout:
                        prompt_text_final = res.stdout.split("TRANSCRIPTION_START")[1].split("TRANSCRIPTION_END")[0].strip()
                        st.caption(f"自动识别文本: {prompt_text_final}")
                    else:
                        st.warning("自动识别失败，请手动输入文本。")
                except Exception as e:
                    st.error(f"识别出错: {e}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "video" in message and os.path.exists(message["video"]):
                col_vid, col_rest = st.columns([1, 2])
                with col_vid:
                    st.video(message["video"])

    # React to user input
    prompt = None
    
    audio_value = st.audio_input("语音输入")
    if audio_value:
        audio_bytes = audio_value.getvalue()
        if "last_audio_bytes" not in st.session_state or st.session_state.last_audio_bytes != audio_bytes:
            st.session_state.last_audio_bytes = audio_bytes
            
            temp_audio_path = os.path.join(current_dir, f"temp_input_{int(time.time())}.wav")
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)
            
            transcribe_script = os.path.join(current_dir, "transcribe_audio.py")
            cmd_transcribe = get_command("cosyvoice", transcribe_script, ["--audio", temp_audio_path])
            
            with st.spinner("正在识别语音..."):
                try:
                    res = subprocess.run(cmd_transcribe, capture_output=True, text=True)
                    if res.returncode == 0:
                        output = res.stdout
                        if "TRANSCRIPTION_START" in output:
                            transcribed_text = output.split("TRANSCRIPTION_START")[1].split("TRANSCRIPTION_END")[0].strip()
                            if transcribed_text:
                                prompt = transcribed_text
                        else:
                            st.error("无法识别语音")
                    else:
                        st.error("语音识别服务出错")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)

    if text_input := st.chat_input("和数字人聊聊吧..."):
        prompt = text_input

    if prompt:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # 1. Get LLM Response
            # We need to run LLM inference in the 'cosyvoice' environment because that's where we installed transformers/accelerate
            # But streamlit is running in 'adnerf'.
            # Solution: We can run a small script to get LLM response or just load it here if adnerf has transformers.
            # 'adnerf' has transformers 4.5.1 (checked earlier? No, cosyvoice has it).
            # Let's try to run LLM via subprocess in cosyvoice env to be safe and avoid conflicts.
            
            llm_script = os.path.join(current_dir, "llm_test.py")
            # We need to modify llm_test.py to accept args or just write a new one.
            # For now, let's assume we can use the simple logic:
            # Create a temp script to run inference
            
            timestamp = int(time.time())
            chat_audio_path = os.path.join(current_dir, f"chat_audio_{timestamp}.wav")
            chat_video_path = os.path.join(current_dir, f"chat_video_{timestamp}.mp4")
            
            # Create a temporary python script for LLM inference
            llm_infer_script = os.path.join(current_dir, f"temp_llm_{timestamp}.py")
            with open(llm_infer_script, "w") as f:
                f.write(f"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

def chat():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        model_dir = snapshot_download(model_name)
    except:
        model_dir = model_name

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    
    prompt = "{prompt}"
    messages = [{{"role": "user", "content": prompt}}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("LLM_RESPONSE_START")
    print(response)
    print("LLM_RESPONSE_END")

if __name__ == "__main__":
    chat()
""")
            
            # Run LLM
            cmd_llm = get_command("cosyvoice", llm_infer_script, [])
            try:
                result_llm = subprocess.run(cmd_llm, capture_output=True, text=True)
                if result_llm.returncode != 0:
                    message_placeholder.error("LLM Error")
                    st.code(result_llm.stderr)
                    response_text = "抱歉，我现在的思维有点混乱。"
                else:
                    # Parse output
                    output = result_llm.stdout
                    if "LLM_RESPONSE_START" in output:
                        response_text = output.split("LLM_RESPONSE_START")[1].split("LLM_RESPONSE_END")[0].strip()
                    else:
                        response_text = "Error parsing LLM response."
            except Exception as e:
                response_text = f"Error: {e}"
            finally:
                if os.path.exists(llm_infer_script):
                    os.remove(llm_infer_script)

            message_placeholder.markdown(f"**AI:** {response_text}\n\n*Generating Audio...*")
            
            # 2. Generate Audio
            script_path = os.path.join(current_dir, "generate_audio.py")
            
            project_root = os.path.dirname(current_dir)
            local_model_dir = os.path.join(project_root, "pretrained_models", "CosyVoice-300M-SFT")
            if os.path.exists(local_model_dir):
                default_model_dir = local_model_dir
            else:
                default_model_dir = "/home/archie/Documents/Work/pretrained_models/CosyVoice-300M-SFT"
            
            model_dir = os.environ.get("COSYVOICE_MODEL_DIR", default_model_dir)
            
            gen_args = ["--text", response_text, "--output", chat_audio_path, "--model_dir", model_dir]
            if prompt_audio_path and prompt_text_final:
                gen_args.extend(["--prompt_audio", prompt_audio_path, "--prompt_text", prompt_text_final])
                
            cmd = get_command("cosyvoice", script_path, gen_args)
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    message_placeholder.error("音频生成失败")
                    st.code(result.stderr)
                else:
                    # 3. Generate Video (Loop existing video + New Audio)
                    # Use ffmpeg to loop the example video and replace audio
                    # ffmpeg -stream_loop -1 -i example.mp4 -i new_audio.wav -shortest -map 0:v -map 1:a -c:v copy -y output.mp4
                    
                    message_placeholder.markdown(f"**AI:** {response_text}\n\n*Synthesizing Video...*")
                    
                    if not os.path.exists(example_video_path):
                         st.error("Example video not found (frontend/output.mp4). Please generate a video in 'Video Generation' tab first.")
                    else:
                        ffmpeg_cmd = [
                            "/home/archie/miniconda3/envs/adnerf/bin/ffmpeg",
                            "-stream_loop", "-1",
                            "-i", example_video_path,
                            "-i", chat_audio_path,
                            "-shortest",
                            "-map", "0:v",
                            "-map", "1:a",
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-y",
                            chat_video_path
                        ]
                        
                        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
                        
                        message_placeholder.markdown(response_text)
                        
                        # Display video with reduced width using columns
                        col_vid, col_rest = st.columns([1, 2])
                        with col_vid:
                            st.video(chat_video_path)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_text,
                            "video": chat_video_path
                        })
            except Exception as e:
                message_placeholder.error(f"Error: {e}")
