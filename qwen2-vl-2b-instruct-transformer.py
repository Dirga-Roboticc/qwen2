from benchmark_transformer import setup_model, prepare_input, run_inference

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

tokenizer, processor, model = setup_model(MODEL_PATH)
messages = prepare_input()

output_text, generated_ids = run_inference(tokenizer, processor, model, messages)

print(generated_ids)
print(output_text)
