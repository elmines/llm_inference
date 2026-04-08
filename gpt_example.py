#!/usr/bin/env python3

import sys
from typing import Sequence, List

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    Conversation,
)

from utils import list_rfind


class GptWrapper:
    def __init__(self, hf_model: str):
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.prompt_token_ids = self.encoding.render_conversation_for_completion(
            Conversation.from_messages(
                [
                    Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                    Message.from_role_and_content(
                        Role.DEVELOPER,
                        DeveloperContent.new().with_instructions(
                            "You are a helpful assistant."
                        ),
                    ),
                ]
            ),
            Role.ASSISTANT,
        )

        self.llm = LLM(model=hf_model, reasoning_parser="openai_gptoss")
        self.tokenizer = self.llm.get_tokenizer()

        added_vocab = self.tokenizer.get_added_vocab()
        self.channel_marker_id = added_vocab["<|channel|>"]
        self.eos_id = added_vocab["<|return|>"]
        self.message_id = added_vocab["<|message|>"]
        self.final_id = self.tokenizer.convert_tokens_to_ids("final")

    def __call__(
        self,
        samples: str | Sequence[str],
        max_tokens: int = 512,
        temperature: float = 1.0,
    ) -> str | List[str]:
        if isinstance(samples, str):
            samples = [samples]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop_token_ids=self.encoding.stop_tokens_for_assistant_actions(),
        )

        batch_prompts: List[TokensPrompt] = [
            {
                "prompt_token_ids": self.prompt_token_ids
                + self.encoding.render_conversation_for_completion(
                    Conversation.from_messages(
                        [Message.from_role_and_content(Role.USER, s)]
                    ),
                    Role.ASSISTANT,
                )
            }
            for s in samples
        ]
        batch_outputs = self.llm.generate(
            batch_prompts, sampling_params=sampling_params
        )

        text_outputs = []
        for output in batch_outputs:
            completion = output.outputs[0]
            output_token_ids = completion.token_ids

            final_token_ids = []
            last_channel_ind = list_rfind(output_token_ids, self.channel_marker_id)
            if output_token_ids[last_channel_ind + 1 : last_channel_ind + 3] == [
                self.final_id,
                self.message_id,
            ]:
                eos_ind = list_rfind(output_token_ids, self.eos_id)
                if eos_ind == -1:
                    final_token_ids = output_token_ids[last_channel_ind + 3 :]
                else:
                    final_token_ids = output_token_ids[last_channel_ind + 3 : eos_ind]
            text_outputs.append(self.tokenizer.decode(final_token_ids))

        if len(text_outputs) == 1:
            return text_outputs[0]
        return text_outputs


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"

    # Run Demo
    gpt = GptWrapper(hf_model=model_path)

    short_question = "How would you bring peace to the Middle East?"
    long_question = """
    What is the opinion of the text below toward the target \"Pineapple on Pizza\"?
    Please answer with only a single word, one of \"favor\", \"against\", or \"neutral\":

    Tangy and savory do not mix.
    """

    samples = [short_question, long_question]

    outputs = gpt(samples)

    print(f"USER: {short_question}")
    print(f"GPT: {outputs[0]}")

    print(f"USER: {long_question}")
    print(f"GPT: {outputs[1]}")
