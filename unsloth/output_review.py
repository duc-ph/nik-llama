import json
import os
import anthropic

# File paths
input_file = "./outputs/checkpoint-800-generation.jsonl"
good_file = "./outputs/checkpoint-800-generation-ok.jsonl"
bad_file = "./outputs/checkpoint-800-generation-bad.jsonl"

# Initialize Claude client
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

# Begin review
with open(input_file, 'r') as infile, \
     open(good_file, 'w') as good_out, \
     open(bad_file, 'w') as bad_out:

    for i, line in enumerate(infile):
        try:
            data = json.loads(line)
            text = data.get("text", "")

            print(f"\n[{i}] Evaluating blog post...")

            if is_post_valid(text):
                print("→ Accepted ✅")
                good_out.write(json.dumps(data) + "\n")
            else:
                print("→ Rejected ❌")
                bad_out.write(json.dumps(data) + "\n")

        except json.JSONDecodeError:
            print(f"[SKIP] Line {i} is not valid JSON.")
        except Exception as e:
            print(f"[ERROR] Unexpected failure on line {i}: {e}")


# import json
# data = []
# with open('checkpoint-1628-generation-bad-regenerated.jsonl', 'r') as f:
#     for line in f:
#         if line.strip():  # skip empty lines
#             data.append(json.loads(line))