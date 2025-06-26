import argparse
import json
import gc
import time
import os
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch
import anthropic

BATCH_SIZE = 2

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_dir", help="Path to model checkpoint")
parser.add_argument("input_path", help="Input JSONL file with blog posts")
parser.add_argument("output_path", help="Where to write the accepted generations")
args = parser.parse_args()

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(args.checkpoint_dir)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
model.eval()

# Load inputs
with open(args.input_path, 'r') as f:
    posts = [json.loads(line)['text'] for line in f]

prompts = []
for post in posts:
    content_marker = '### Content: \n'
    content_start_idx = post.find(content_marker) + len(content_marker)
    first_paragraph_end = post.find('\n', content_start_idx) + 1
    prompts.append(post[:first_paragraph_end])

# Claude API
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def is_post_valid(post_text):
    """Claude-based evaluation for blog post quality."""
    prompt = (
        "You are a content quality evaluator. Given the following blog post, determine if it is well-structured, "
        "coherent, and free from anomalies such as endless character streams or abrupt transitions. A common error is that paragraphs shouldn't start with \u2019t. e.g. 'their cups.\n\u2019t a small detail' means a bad post.\n\n"
        f"Blog Post:\n{post_text}\n\n"
        "Respond with 'Yes' if the post is acceptable, or 'No' if it should be rejected."
    )
    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1,
            messages=[{"role": "user", "content": prompt}]
        )
        decision = response.content[0].text.strip().lower()
        return decision.startswith("yes")
    except Exception as e:
        print(f"[ERROR] Claude evaluation failed: {e}")
        return False  # Treat uncertain cases as invalid

# Main inference loop
with open(args.output_path, 'w') as out_file:
    while prompts:
        failed_prompts = []
        for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch_prompts, return_tensors='pt',
                            padding=True, truncation=True).to('cuda')

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=2000)
            end_time = time.time()

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for prompt, generated in zip(batch_prompts, decoded):
                if is_post_valid(generated):
                    json.dump({"text": generated.strip()}, out_file)
                    out_file.write('\n')
                    out_file.flush()
                else:
                    # Add back to backlog for retry
                    failed_prompts.append(prompt)
                    print(f'Post failed. Total failed: {len(failed_prompts)}')

            generated_tokens = sum(len(tokenizer.encode(d)) for d in decoded)
            elapsed_time = end_time - start_time
            throughput = generated_tokens / elapsed_time if elapsed_time > 0 else float('inf')
            print(f"Batch {i//BATCH_SIZE + 1}: {generated_tokens} tokens in {elapsed_time:.2f} sec "
                f"({throughput:.2f} tokens/sec)")

            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
        
        # Re-generate failed prompts
        prompts = failed_prompts[:]
        failed_prompts = []
