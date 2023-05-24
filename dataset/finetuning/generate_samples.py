import random
import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = """
Your task is to generate idiomatic snippets of complete Go files that will be used to fine-tune a Code LLM. 

- The snippets must reflect the full range of projects people develop using Go.
- The snippets must vary in length and complexity and in nature.
- The snippets must reflect industry usage of Go in real-world projects.
- The snippets must not be dummy examples that would not be found in actual projects.
- The snippets must display idiomatic usage of Go through best practices and common patterns.
- The snippets must show how to use the standard library and common third-party libraries.

Output only the code, one sample per message.
"""

example_prompt = """
New Snippet
"""

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

func main() {
\thttp.HandleFunc("/posts", createPost)
\thttp.HandleFunc("/posts/", getPost)

\tlog.Fatal(http.ListenAndServe(":8080", nil))
}
"""

conversation = [{"role": "system", "content": system_prompt},
        {"role": "user", "content": example_prompt},
        {"role": "assistant", "content": example_response},
        {"role": "user", "content": example_prompt}]

with open('dataset/finetuning/ai-samples.jsonl', 'w') as f:
    for i in range(1, 10):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            # Random temperate between 0.5 and 1.5
            temperature=random.uniform(0.5, 1.5),
        )

        # Extract the choices (samples) from the response
        samples = [choice['message']['content'].strip() for choice in response.choices]
        print("Generated:", samples[0])

        # Save the samples to a JSONL file
        for sample in samples:
            f.write(json.dumps({"sample": sample}) + os.linesep)
            f.flush()
