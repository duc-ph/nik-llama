import argparse
import json
import torch
from unsloth import FastLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_dir")
parser.add_argument("output_path")
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(args.checkpoint_dir)

with open('../data/val_data_2025_onward.jsonl', 'r') as f:
    posts = [json.loads(line)['text'] for line in f]

prompts = []
for post in posts:
    content_marker = '### Content: \n'
    content_start_idx = post.find(content_marker) + len(content_marker)
    first_paragraph_end = post.find('\n', content_start_idx) + 1
    prompts.append(post[:first_paragraph_end])

results = []
BATCH_SIZE = 64
for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i + BATCH_SIZE]
    inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results.extend(decoded)

with open(args.output_path, 'w') as out_file:
    for line in results:
        out_file.write(line.strip() + '\n')