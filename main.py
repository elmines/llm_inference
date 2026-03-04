from vllm import LLM

def main():
    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate("Hello, my name is")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {generated_text}")


if __name__ == "__main__":
    main()
