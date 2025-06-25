import argparse
import json
import gc
import time
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch

BATCH_SIZE = 2

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_dir")
parser.add_argument("output_path")
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(args.checkpoint_dir)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
model.eval()

with open('../data/val_data_2025_onward.jsonl', 'r') as f:
    posts = [json.loads(line)['text'] for line in f]

prompts = []
for post in posts:
    content_marker = '### Content: \n'
    content_start_idx = post.find(content_marker) + len(content_marker)
    first_paragraph_end = post.find('\n', content_start_idx) + 1
    prompts.append(post[:first_paragraph_end])

with open(args.output_path, 'w') as out_file:
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_prompts, return_tensors='pt',
                           padding=True, truncation=True).to('cuda')

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=4000)
        end_time = time.time()

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Write to file immediately
        for line in decoded:
            json.dump({"text": line.strip()}, out_file)
            out_file.write('\n')

        generated_tokens = sum(len(tokenizer.encode(d)) for d in decoded)
        elapsed_time = end_time - start_time
        throughput = generated_tokens / elapsed_time if elapsed_time > 0 else float('inf')
        print(f"Batch {i//BATCH_SIZE + 1}: {generated_tokens} tokens in {elapsed_time:.2f} sec "
              f"({throughput:.2f} tokens/sec)")

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
