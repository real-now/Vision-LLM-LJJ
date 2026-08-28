"""
[음성 + LLM 프로세스]  voice_llm_held_object.py

마이크를 짧은 구간으로 계속 녹음하면서 Whisper로 받아쓰고,
질문 키워드가 들리면 Vision 프로세스에 ROI 캡처를 요청한 뒤,
돌아온 이미지를 Gemma 멀티모달로 판단하고 Piper TTS로 대답한다.

    마이크 -> Whisper -> (질문 감지) -> request.json
           -> roi.jpg + roi_meta.json -> Gemma -> Piper

실행 (vision_on_request.py와 다른 터미널에서):
    source .venv/bin/activate
    python voice_llm_held_object.py

종료: Ctrl + C
"""

import base64
import json
import os
import subprocess
import time
import wave

import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

# ============================ 설정 ============================

# --- Gemma ---
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
CONTEXT_WINDOW = 2048
MAX_TOKENS = 60

# --- 프로세스 간 파일 ---
REQUEST_PATH = "src/output/request.json"
TEMP_REQUEST_PATH = "src/output/request_tmp.json"
ROI_IMAGE_PATH = "src/output/roi.jpg"
ROI_META_PATH = "src/output/roi_meta.json"
ROI_TIMEOUT = 5.0  # Vision 응답 대기 한계 (초)

# --- STT (whisper.cpp) ---
MIC_DEVICE = "plughw:3,0"
CHUNK_FILE = "src/audio/chunk.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"
CHUNK_SECONDS = 3  # 한 번에 녹음할 길이

# 무음 판정: 이 값보다 조용하면 Whisper를 아예 돌리지 않는다 (CPU 절약)
RMS_THRESHOLD = 300

# --- TTS (Piper) ---
PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"
SPEAKER_DEVICE = "plughw:3,0"
TTS_COOLDOWN = 1.0  # 재생 직후 이 시간만큼은 녹음하지 않는다

# --- 질문 감지 ---
# 이 중 하나라도 들어 있으면 질문으로 본다.
TRIGGER_KEYWORDS = [
    "뭐", "뭘", "무엇", "무슨", "이게", "이거", "들고", "손에",
]

# Whisper가 무음 구간에서 자주 만들어내는 환청 문장들
IGNORE_PHRASES = [
    "시청해", "구독", "감사합니다", "다음 영상", "안녕하세요",
    "MBC", "KBS", "자막",
]


# ============================ STT ============================


def record_chunk():
    subprocess.run(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(CHUNK_SECONDS),
            CHUNK_FILE,
        ],
        check=True,
        capture_output=True,
    )


def chunk_rms():
    """녹음된 구간의 평균 음량. 무음이면 Whisper를 건너뛰기 위해 쓴다."""
    with wave.open(CHUNK_FILE, "rb") as wav:
        frames = wav.readframes(wav.getnframes())

    if not frames:
        return 0.0

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    return float(np.sqrt(np.mean(samples ** 2)))


def transcribe():
    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", CHUNK_FILE,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    lines = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue

        lines.append(line)

    return " ".join(lines)


def is_question(text):
    if not text:
        return False

    for phrase in IGNORE_PHRASES:
        if phrase in text:
            return False

    for keyword in TRIGGER_KEYWORDS:
        if keyword in text:
            return True

    return False


# ============================ TTS ============================


def text_to_speech(text):
    for char in "*_`#\n\r":
        text = text.replace(char, " ")

    text = " ".join(text.split())

    if not text:
        return

    subprocess.run(
        [
            PIPER_PYTHON,
            "-m",
            "piper",

            "-m",
            PIPER_MODEL,

            "-f",
            OUTPUT_FILE,

            "--",
            text,
        ],
        check=True,
    )

    subprocess.run(
        [
            "aplay",
            "-D",
            SPEAKER_DEVICE,
            OUTPUT_FILE,
        ],
        check=True,
    )


# ==================== Vision 프로세스와의 핸드셰이크 ====================


def request_roi(question):
    """Vision 프로세스에 '지금 화면을 캡처해 달라'고 요청한다."""
    with open(TEMP_REQUEST_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {"question": question, "timestamp": time.time()},
            file,
            ensure_ascii=False,
        )

    os.replace(TEMP_REQUEST_PATH, REQUEST_PATH)


def wait_for_roi(previous_mtime):
    """Vision이 새 ROI를 저장할 때까지 기다린다. 실패하면 None."""
    deadline = time.time() + ROI_TIMEOUT

    while time.time() < deadline:
        if os.path.exists(ROI_META_PATH):
            current = os.path.getmtime(ROI_META_PATH)

            if current != previous_mtime:
                with open(ROI_META_PATH, "r", encoding="utf-8") as file:
                    meta = json.load(file)

                return current, meta

        time.sleep(0.05)

    return previous_mtime, None


# ============================ Prompt ============================


def build_messages(image_data, meta, question):
    if meta["mode"] == "object":
        system_content = """
                          Instruction:
                          주어진 이미지는 사람이 손에 들고 있는 물건을 잘라낸 것이다.
                          사용자 질문에 답하시오.

                          Constraint:
                          참고 정보로 주어진 후보 이름은 틀릴 수 있으므로,
                          반드시 이미지를 우선하여 판단하시오.
                          이미지에서 확실하게 알아볼 수 없으면
                          "잘 모르겠습니다"라고 답하시오.
                          응답은 음성으로 재생되므로 기호나 목록을 쓰지 마시오.

                          Output Format:
                          한국어 한 문장.
                         """

        user_text = (
            f"참고 정보: 객체 탐지 후보 이름은 {meta['hint']}입니다.\n"
            f"질문: {question}"
        )
    else:
        system_content = """
                          Instruction:
                          주어진 이미지는 카메라에 잡힌 사람이다.
                          이 사람이 손에 들고 있는 물건을 보고 사용자 질문에 답하시오.

                          Constraint:
                          손에 들고 있는 물건만 답하고, 배경의 물건은 무시하시오.
                          아무것도 들고 있지 않으면 "아무것도 들고 있지 않습니다"라고 답하시오.
                          이미지에서 확실하게 알아볼 수 없으면
                          "잘 모르겠습니다"라고 답하시오.
                          응답은 음성으로 재생되므로 기호나 목록을 쓰지 마시오.

                          Output Format:
                          한국어 한 문장.
                         """

        user_text = f"질문: {question}"

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    },
                },
            ],
        },
    ]


def ask_gemma(llm, meta, question):
    with open(ROI_IMAGE_PATH, "rb") as file:
        image_bytes = file.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data = "data:image/jpeg;base64," + image_base64

    response = llm.create_chat_completion(
        messages=build_messages(image_data, meta, question),
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )

    return response["choices"][0]["message"]["content"].strip()


# ============================ main ============================

os.makedirs("src/audio", exist_ok=True)
os.makedirs("src/output", exist_ok=True)

chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

print("Gemma 로드 완료.")
print("듣고 있습니다. (예: \"이게 뭐야?\", \"내가 들고 있는 게 뭐야?\")")
print("종료: Ctrl + C\n")

if os.path.exists(ROI_META_PATH):
    roi_mtime = os.path.getmtime(ROI_META_PATH)
else:
    roi_mtime = 0

try:
    while True:
        # 1. 짧게 녹음
        record_chunk()

        # 2. 무음이면 Whisper를 돌리지 않는다
        if chunk_rms() < RMS_THRESHOLD:
            continue

        # 3. 받아쓰기
        text = transcribe()

        if not text:
            continue

        print(f"[들림] {text}")

        # 4. 질문인지 판단
        if not is_question(text):
            continue

        print(f"\n[질문 감지] {text}")

        # 5. Vision에 ROI 캡처 요청
        request_roi(text)

        roi_mtime, meta = wait_for_roi(roi_mtime)

        if meta is None:
            print("[오류] Vision 프로세스 응답이 없습니다.\n")
            continue

        if meta["mode"] == "none":
            answer = "카메라에 사람이 보이지 않습니다."
        else:
            print(f"[ROI] mode={meta['mode']}  hint={meta['hint']}")

            start = time.time()
            answer = ask_gemma(llm, meta, text)
            print(f"[Gemma] ({time.time() - start:.1f}초)")

        print(f"{answer}\n")

        # 6. 음성 출력 후 잠시 쉬어 자기 목소리를 다시 듣지 않게 한다
        text_to_speech(answer)
        time.sleep(TTS_COOLDOWN)

except KeyboardInterrupt:
    print("\n음성 프로세스를 종료했습니다.")
