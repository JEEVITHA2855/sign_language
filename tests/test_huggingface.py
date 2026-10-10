import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from .env")

if not HF_MODEL:
    raise ValueError("HF_MODEL is missing from .env")


# ============================================================
# CREATE HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


# ============================================================
# INPUT
# ============================================================

raw_sentence = "I WANT WATR"


# ============================================================
# PROMPT
# ============================================================

system_prompt = """
You are a sign-language transcription correction system.

Your ONLY job is to correct:
- spelling
- capitalization
- basic grammar

STRICT RULES:
- Do not add information.
- Do not invent words.
- Do not change the meaning.
- Do not explain anything.
- Do not show reasoning.
- Return ONLY the corrected sentence.
"""


user_prompt = f"""
/no_think

Correct this sign-language transcription:

{raw_sentence}

Return ONLY the corrected sentence.
"""


# ============================================================
# CALL MODEL
# ============================================================

try:

    response = client.chat.completions.create(
        model=HF_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        # Give enough room for a response
        max_tokens=100,

        # Qwen recommends non-thinking mode around these settings
        temperature=0.7,

        top_p=0.8,

        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )


    # ========================================================
    # EXTRACT RESULT
    # ========================================================

    message = response.choices[0].message

    print("\n" + "=" * 60)
    print("FULL RESPONSE")
    print("=" * 60)
    print(response)

    print("\n" + "=" * 60)
    print("CONTENT")
    print("=" * 60)
    print(repr(message.content))

    print("\n" + "=" * 60)
    print("REASONING")
    print("=" * 60)

    if hasattr(message, "reasoning_content"):
        print(repr(message.reasoning_content))
    else:
        print("No reasoning field")


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    if message.content:

        result = message.content.strip()

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(result)

    else:

        print("\nNo final content returned.")

        # If provider still returns reasoning, show it
        if hasattr(message, "reasoning_content"):

            reasoning = message.reasoning_content

            if reasoning:
                print("\nThe model still used reasoning mode.")
                print("Reasoning returned:")
                print(reasoning)


except Exception as e:

    print("\n" + "=" * 60)
    print("ERROR")
    print("=" * 60)

    print(type(e).__name__)
    print(str(e))