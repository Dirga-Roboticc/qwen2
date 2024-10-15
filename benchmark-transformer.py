import time
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

def setup_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cuda",
    )
    return tokenizer, processor, model

def prepare_input():
    return [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "./tes.png",
                    },
                    {
                        "type": "text",
                        "text": "find search icon of the left sidebar of visual studio code and Generate the caption in English with grounding:",
                    },
                ],
            },
        ],
    ]

def run_inference(tokenizer, processor, model, messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    output_text = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return output_text, generated_ids

def benchmark(num_runs=10):
    tokenizer, processor, model = setup_model(MODEL_PATH)
    messages = prepare_input()

    # Warm-up run (not counted in the benchmark)
    _ = run_inference(tokenizer, processor, model, messages)

    total_time = 0
    for i in range(num_runs):
        start_time = time.time()
        output_text, generated_ids = run_inference(tokenizer, processor, model, messages)
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        print(f"Run {i+1}: Inference time: {inference_time:.4f} seconds")
        
        if i == 0:
            print("\nOutput of the first run:")
            print(output_text)

    average_time = total_time / num_runs
    print(f"\nAverage inference time over {num_runs} runs: {average_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
