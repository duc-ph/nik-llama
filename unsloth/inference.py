"""Inference script for generating blog posts with quality evaluation.

Generates blog posts from fine-tuned model checkpoints and uses Claude
for quality assessment to filter out low-quality generations.
"""

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


def is_post_valid(post_text: str, client: anthropic.Anthropic) -> bool:
    """Evaluate blog post quality using Claude.
    
    Args:
        post_text: Generated blog post content
        client: Anthropic API client
        
    Returns:
        bool: True if post passes quality evaluation
    """
    prompt = (
        "You are a content quality evaluator. Given the following blog post, determine if it is well-structured, "
        "coherent, and free from anomalies such as endless character streams or abrupt transitions. A common error is that paragraphs shouldn't start with 't. e.g. 'their cups.\n't a small detail' means a bad post.\n\n"
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
        return False


def extract_prompts(posts: list[str]) -> list[str]:
    """Extract prompts from blog posts for generation.
    
    Args:
        posts: List of blog post texts
        
    Returns:
        List of extracted prompts
    """
    prompts = []
    for post in posts:
        content_marker = '### Content: \n'
        content_start_idx = post.find(content_marker) + len(content_marker)
        first_paragraph_end = post.find('\n', content_start_idx) + 1
        prompts.append(post[:first_paragraph_end])
    return prompts


def generate_batch(model, tokenizer, batch_prompts: list[str]) -> tuple[list[str], float]:
    """Generate text for a batch of prompts.
    
    Args:
        model: Fine-tuned language model
        tokenizer: Model tokenizer
        batch_prompts: List of prompts to generate from
        
    Returns:
        Tuple of (generated texts, generation time)
    """
    inputs = tokenizer(
        batch_prompts, 
        return_tensors='pt',
        padding=True, 
        truncation=True
    ).to('cuda')

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2000)
    end_time = time.time()

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    return decoded, end_time - start_time


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Generate blog posts from fine-tuned model checkpoint"
    )
    parser.add_argument("checkpoint_dir", help="Path to model checkpoint")
    parser.add_argument("input_path", help="Input JSONL file with blog posts")
    parser.add_argument("output_path", help="Where to write accepted generations")
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(args.checkpoint_dir)
    FastLanguageModel.for_inference(model)
    model.eval()

    print(f"Loading input data from: {args.input_path}")
    with open(args.input_path, 'r') as f:
        posts = [json.loads(line)['text'] for line in f]

    prompts = extract_prompts(posts)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    print(f"Starting inference with {len(prompts)} prompts")
    print(f"Output will be saved to: {args.output_path}")
    
    with open(args.output_path, 'w') as out_file:
        while prompts:
            failed_prompts = []
            
            for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating posts"):
                batch_prompts = prompts[i:i + BATCH_SIZE]
                decoded, elapsed_time = generate_batch(model, tokenizer, batch_prompts)

                for prompt, generated in zip(batch_prompts, decoded):
                    if is_post_valid(generated, client):
                        json.dump({"text": generated.strip()}, out_file)
                        out_file.write('\n')
                        out_file.flush()
                    else:
                        failed_prompts.append(prompt)
                        print(f'Post failed quality check. Total failed: {len(failed_prompts)}')

                generated_tokens = sum(len(tokenizer.encode(d)) for d in decoded)
                throughput = generated_tokens / elapsed_time if elapsed_time > 0 else float('inf')
                print(f"Batch {i//BATCH_SIZE + 1}: {generated_tokens} tokens in {elapsed_time:.2f}s "
                    f"({throughput:.2f} tokens/s)")
            
            prompts = failed_prompts[:]
            if prompts:
                print(f"Retrying {len(prompts)} failed prompts...")
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()