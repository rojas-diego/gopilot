import time
import logging
import random
import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def new_prompt(description, package): return f"New Snippet. Description: {description} Package: {package}"

system_prompt = """
Your task is to generate idiomatic snippets of complete Go files based on program descriptions.

- The snippets must 
    - Reflect the full range of projects people develop using Go.
    - Vary in length and complexity and in nature.
    - Reflect industry usage of Go in real-world projects.
    - Must show how to use the common third-party libraries.
    - Must not be from the `main` package, but rather a diverse set of packages.
    - May or may not be a test file.
    - Should not be a single function or a single line of code.

Output only the code, one sample per message.
"""

example_prompt = new_prompt("A HTTP server that manages posts", "posts")

example_response = """
package posts

import (
\t"fmt"
\t"log"
\t"net/http"
\t"models"
)

func createPost(w http.ResponseWriter, r *http.Request) {
\tif r.Method != "POST" {
\t\tw.WriteHeader(http.StatusMethodNotAllowed)
\t\treturn
\t}

\tif err := r.ParseForm(); err != nil {
\t\tlog.Println(err)
\t\tw.WriteHeader(http.StatusBadRequest)
\t\treturn
\t}

\ttitle := r.FormValue("title")
\tcontent := r.FormValue("content")

\tpost := models.Post{
\t\tTitle:   title,
\t\tContent: content,
\t}

\tif err := post.Create(); err != nil {
\t\tlog.Println(err)
\t\tw.WriteHeader(http.StatusInternalServerError)
\t\treturn
\t}

\tw.WriteHeader(http.StatusCreated)
\tfmt.Fprintf(w, "Post created successfully")
}

func getPost(w http.ResponseWriter, r *http.Request) {
\tif r.Method != "GET" {
\t\tw.WriteHeader(http.StatusMethodNotAllowed)
\t\treturn
\t}

\tid := r.URL.Query().Get("id")

\tpost, err := models.GetPost(id)
\tif err != nil {
\t\tlog.Println(err)
\t\tw.WriteHeader(http.StatusInternalServerError)
\t\treturn
\t}

\tw.WriteHeader(http.StatusOK)
\tfmt.Fprintf(w, "Post: %v", post)
}
"""


def new_conversations(description, package): return [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": example_prompt},
    {"role": "assistant", "content": example_response},
    {"role": "user", "content": new_prompt(description, package)}
]

num_lines_to_skip = 0
with open('dataset/finetuning/programs-from-descriptions.jsonl', 'r') as f:
    for _ in f:
        num_lines_to_skip += 1


with open('dataset/finetuning/programs-from-descriptions.jsonl', 'a') as f:
    with open('dataset/finetuning/programs-from-descriptions-prompts.jsonl', 'r') as prompts:
        for _ in range(num_lines_to_skip):
            next(prompts)

        print(f"Skipping {num_lines_to_skip} prompts")

        # Generate samples for the remaining prompts
        for line in prompts:
            prompt = json.loads(line)
            # Rest of the code here
            description = prompt["description"]
            package = prompt["package"]
            temperature = random.uniform(0.2, 0.6)
            print(f"Generating one sample for package '{package}' with description '{description}' and temperature {temperature:.2f}")

            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=new_conversations(description, package),
                        temperature=temperature,
                    )

                    # Extract the choices (samples) from the response
                    samples = [choice['message']['content'].strip() for choice in response.choices]  # type: ignore

                    # Save the samples to a JSONL file
                    for sample in samples:
                        print("--- Sample ---")
                        print(sample)
                        f.write(json.dumps({"sample": sample}) + os.linesep)
                        f.flush()
                    break
                except Exception as e:
                    print("Encountered an error, waiting 3 seconds and retrying...", e)
                    time.sleep(3)
