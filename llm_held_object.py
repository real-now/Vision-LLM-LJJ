"""
[LLM 프로세스]  llm_held_object.py

vision_held_object.py가 저장한 ROI 이미지를 감시하다가,
새 이미지가 들어오면 Gemma 멀티모달로 "무엇을 들고 있는지" 판단하고
Piper TTS로 음성 출력한다.

실행 (vision_held_object.py와 다른 터미널에서):
    source .venv/bin/activate
    python llm_held_object.py
"""

import base64
import json
import os
import subprocess
import time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

# ============================ 설정 ============================

GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"

ROI_IMAGE_PATH = "src/output/roi.jpg"
ROI_META_PATH = "src/output/roi_meta.json"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 60

# --- TTS (Piper) ---
PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"
SPEAKER_DEVICE = "plughw:3,0"


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


# ============================ Prompt ============================


def build_messages(image_data, meta):
    if meta["mode"] == "object":
        # YOLO가 물건 위치를 짚어준 경우: 잘라낸 물건 이미지 + 클래스 힌트
        system_content = """
                          Instruction:
                          주어진 이미지는 사람이 손에 들고 있는 물건을 잘라낸 것이다.
                          이 물건이 무엇인지 판단하시오.

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
            "이 사람이 들고 있는 물건이 무엇인지 말하시오."
        )
    else:
        # YOLO 클래스에 없는 물건: 사람 전체 이미지에서 Gemma가 직접 찾는다
        system_content = """
                          Instruction:
                          주어진 이미지는 카메라에 잡힌 사람이다.
                          이 사람이 손에 들고 있는 물건이 무엇인지 판단하시오.

                          Constraint:
                          손에 들고 있는 물건만 답하고, 배경의 물건은 무시하시오.
                          아무것도 들고 있지 않으면 "아무것도 들고 있지 않습니다"라고 답하시오.
                          이미지에서 확실하게 알아볼 수 없으면
                          "잘 모르겠습니다"라고 답하시오.
                          응답은 음성으로 재생되므로 기호나 목록을 쓰지 마시오.

                          Output Format:
                          한국어 한 문장.
                         """

        user_text = "이 사람이 손에 들고 있는 물건이 무엇인지 말하시오."

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


# ============================ main ============================

os.makedirs("src/audio", exist_ok=True)

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
print("Vision 프로세스의 ROI 입력 대기 중...\n")

# 메타 JSON의 수정 시각만 감시한다. (이미지가 항상 먼저 저장되므로 안전)
if os.path.exists(ROI_META_PATH):
    last_modified = os.path.getmtime(ROI_META_PATH)
else:
    last_modified = 0

while True:
    if os.path.exists(ROI_META_PATH):
        current_modified = os.path.getmtime(ROI_META_PATH)

        if current_modified != last_modified:
            last_modified = current_modified

            with open(ROI_META_PATH, "r", encoding="utf-8") as file:
                meta = json.load(file)

            with open(ROI_IMAGE_PATH, "rb") as file:
                image_bytes = file.read()

            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_data = "data:image/jpeg;base64," + image_base64

            print(f"[ROI] mode={meta['mode']}  hint={meta['hint']}")

            start = time.time()

            response = llm.create_chat_completion(
                messages=build_messages(image_data, meta),
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )

            answer = response["choices"][0]["message"]["content"].strip()

            print(f"\n[Gemma] ({time.time() - start:.1f}초)")
            print(answer + "\n")

            text_to_speech(answer)

    time.sleep(0.1)
