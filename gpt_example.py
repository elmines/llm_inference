#!/usr/bin/env python3

from typing import Sequence, List

from vllm import LLM, SamplingParams
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    Conversation
)

from utils import list_rfind

class GptWrapper:
    def __init__(self, hf_model: str | LLM):
        self.llm = LLM(model=hf_model, reasoning_parser="openai_gptoss")
        self.tokenizer = self.llm.get_tokenizer()

        added_vocab = self.tokenizer.get_added_vocab()
        self.channel_marker_id = added_vocab['<|channel|>']
        self.eos_id = added_vocab['<|return|>']
        self.message_id = added_vocab['<|message|>']
        self.final_id = self.tokenizer.convert_tokens_to_ids('final')

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.sampling_params = SamplingParams(max_tokens=384,
                                              temperature=1,
                                              stop_token_ids=self.encoding.stop_tokens_for_assistant_actions())
    
    def __call__(self, convos: Conversation | Sequence[Conversation]) -> str | List[str]:
        if isinstance(convos, Conversation):
            convos = [convos]

        batch_prompts = [
            {"prompt_token_ids": self.encoding.render_conversation_for_completion(c, Role.ASSISTANT)
            } for c in convos
        ]
        batch_outputs = self.llm.generate(batch_prompts, sampling_params=self.sampling_params)

        text_outputs = []
        for output in batch_outputs:
            completion = output.outputs[0]
            output_token_ids = completion.token_ids

            final_token_ids = []
            last_channel_ind = list_rfind(output_token_ids, self.channel_marker_id)
            if output_token_ids[last_channel_ind + 1: last_channel_ind + 3] == [self.final_id, self.message_id]:
                eos_ind = list_rfind(output_token_ids, self.eos_id)
                if eos_ind == -1:
                    final_token_ids = output_token_ids[last_channel_ind + 3:]
                else:
                    final_token_ids = output_token_ids[last_channel_ind + 3 : eos_ind]
            text_outputs.append(self.tokenizer.decode(final_token_ids))
        
        if len(text_outputs) == 1:
            return text_outputs[0]
        return text_outputs