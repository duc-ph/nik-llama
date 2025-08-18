"""Post-generation quality review script.

Reviews generated blog posts using Claude to separate high-quality
generations from those that need improvement or regeneration.
"""

import argparse
import json
import os
import anthropic


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


def main():
    """Main review function."""
    parser = argparse.ArgumentParser(
        description="Review generated blog posts for quality"
    )
    parser.add_argument("input_file", help="Input JSONL file with generations")
    parser.add_argument("good_file", help="Output file for accepted posts")
    parser.add_argument("--bad_file", help="Output file for rejected posts")
    args = parser.parse_args()

    bad_file = args.bad_file or args.good_file.replace("-ok.jsonl", "-bad.jsonl")
    
    print(f"Reviewing posts from: {args.input_file}")
    print(f"Good posts will be saved to: {args.good_file}")
    print(f"Bad posts will be saved to: {bad_file}")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    accepted_count = 0
    rejected_count = 0

    with open(args.input_file, 'r') as infile, \
         open(args.good_file, 'w') as good_out, \
         open(bad_file, 'w') as bad_out:

        for i, line in enumerate(infile):
            try:
                data = json.loads(line)
                text = data.get("text", "")

                print(f"[{i+1}] Evaluating blog post...")

                if is_post_valid(text, client):
                    print("→ Accepted ✅")
                    good_out.write(json.dumps(data) + "\n")
                    accepted_count += 1
                else:
                    print("→ Rejected ❌")
                    bad_out.write(json.dumps(data) + "\n")
                    rejected_count += 1

            except json.JSONDecodeError:
                print(f"[SKIP] Line {i+1} is not valid JSON")
                rejected_count += 1
            except Exception as e:
                print(f"[ERROR] Unexpected failure on line {i+1}: {e}")
                rejected_count += 1

    print(f"\nReview completed!")
    print(f"Accepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Total: {accepted_count + rejected_count}")


if __name__ == "__main__":
    main()