# Querying a database using natural language

This sample demonstrates querying a MySQL database using natural language thanks to [LangChain](https://www.langchain.com/), [Chainlit](https://docs.chainlit.io/get-started/overview), and [Ollama](https://ollama.com/).

Runs locally. No internet.

![Demo](app/demo.gif)

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Visual Studio Code](https://code.visualstudio.com/) + [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- A dedicated NVIDIA GPU 

## Running locally

1. Open the repository in Visual Studio Code.
2. Use the Dev Containers command `Reopen in a Container`.
    - Tip: Use `Ctrl` + `Shift` + `P` to open the command pallete.
3. Install a LLM in the `llm` container using Ollama.
    - Command: `ollama pull mistral:7b-instruct-v0.2-q5_K_M`
4. Set the `LLM_MODEL` in the `.env` file (If you choose a different model).
5. When the Dev Container has loaded, launch the app using `F5`.
