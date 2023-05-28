import random
import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

packages = [
    "middleware",
    "models",
    "routes",
    "services",
    "utils",
    "transactions",
    "validation",
    "views",
    "controllers",
    "config",
    "db",
    "handlers",
    "helpers",
    "migrations",
    "raft",
    "scripts",
    "templates",
    "ssh",
    "storage",
    "pathfinding",
    "tcp",
    "udp",
    "http",
    "websocket",
    "rpc",
    "json",
    "xml",
    "yaml",
    "csv",
    "sql",
    "redis",
    "mongodb",
    "postgresql",
    "mysql",
    "sqlite",
    "cassandra",
    "aws",
    "gcp",
    "azure",
    "kubernetes",
    "docker",
    "helm",
    "terraform",
    "ansible",
    "puppet",
    "chef",
    "saltstack",
    "git",
    "github",
    "gitlab",
    "bitbucket",
    "jira",
    "slack",
    "discord",
    "zoom",
    "google",
    "sort",
    "search",
    "filter",
    "mapreduce",
    "zookeeper",
    "kafka",
    "rabbitmq",
    "nats",
    "grpc",
    "protobuf",
    "jwt",
    "oauth",
    "prometheus",
    "grafana",
    "elasticsearch",
    "kibana",
    "logstash",
    "fluentd",
    "datadog",
    "parser",
]

system_prompt = """
Your task is to generate idiomatic snippets of complete Go files that will be used to fine-tune a Code LLM. 

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


def new_example_prompt(package): return "New Snippet. Package: " + package


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


def new_example_conversation(package): return [{"role": "system", "content": system_prompt},
                                               {"role": "user", "content": new_example_prompt("posts")},
                                               {"role": "assistant", "content": example_response},
                                               {"role": "user", "content": new_example_prompt(package)}]


with open('dataset/finetuning/ai-samples.jsonl', 'a') as f:
    for i in range(1, 10):
        package = random.choice(packages)
        package = package if random.random() < 0.8 else package + "_test"
        temperature = random.uniform(0.3, 1.2)

        print("Generating 2 sample for package '" + package + "' with temperature " + str(temperature))

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=new_example_conversation(package),
            temperature=temperature,
            n=2,
        )

        # Extract the choices (samples) from the response
        samples = [choice['message']['content'].strip() for choice in response.choices]  # type: ignore

        # Save the samples to a JSONL file
        for sample in samples:
            print("--- Sample ---")
            print(sample)
            f.write(json.dumps({"sample": sample}) + os.linesep)
            f.flush()
