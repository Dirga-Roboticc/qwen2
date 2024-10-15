import time
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

def setup_model():
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 10, "video": 10},
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return llm, sampling_params, processor

def prepare_input():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "./tes.png",
                },
                {"type": "text", "text": "find search icon of the left sidebar of visual studio code and Generate with grounding:"},
            ],
        },
    ]
    return messages

def run_inference(llm, sampling_params, processor, messages):
    prompt = processor.apply_chat_template(messages, tokenize=False)
    image_inputs, video_inputs = process_vision_info(messages)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    start_time = time.time()
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    end_time = time.time()

    return end_time - start_time

def benchmark(num_runs=10):
    llm, sampling_params, processor = setup_model()
    messages = prepare_input()

    # Warm-up run (not counted in the benchmark)
    _ = run_inference(llm, sampling_params, processor, messages)

    total_time = 0
    for _ in range(num_runs):
        inference_time = run_inference(llm, sampling_params, processor, messages)
        total_time += inference_time
        print(f"Inference time: {inference_time:.4f} seconds")

    average_time = total_time / num_runs
    print(f"\nAverage inference time over {num_runs} runs: {average_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
