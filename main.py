import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

INPUT_FILE = "prompts.txt"
OUTPUT_FILE = "responses.json"
MODEL_NAME = "openai/gpt-oss-120b"


def read_prompts(file_path):
    """Read prompts from a text file, one per line."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def get_llm_response(prompt):
    """Send one prompt to the LLM and return its reply."""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def save_responses(results, file_path):
    """Save the results list into a JSON file."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


def main():
    prompts = read_prompts(INPUT_FILE)
    results = []
    for number, prompt in enumerate(prompts, start=1):
        print(f"Processing {number}/{len(prompts)}...")
        response = get_llm_response(prompt)
        results.append({"id": number, "prompt": prompt, "response": response})
    save_responses(results, OUTPUT_FILE)
    print("Done! Check responses.json")


if __name__ == "__main__":
    main()