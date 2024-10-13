from locust import task, FastHttpUser, events, tag
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import List,Tuple
from transformers import AutoTokenizer
import threading

lock = threading.Lock()
prompt_id=0

@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm"])
    parser.add_argument("--endpoint", type=str, default="/v1/completions", help="API endpoint.")
    parser.add_argument("--dataset", type=str, default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the next release.")
    parser.add_argument("--dataset-name", type=str, default="random",
        choices=["random"], help="Name of the dataset to benchmark on.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument("--model", type=str,required=True, help="Name of the model.")
    parser.add_argument("--tokenizer", type=str, help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument("--best-of", type=int, default=1,
        help="Generates `best_of` sequences per prompt and returns the best one.")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
    parser.add_argument("--logprobs", type=int, default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )   
    parser.add_argument("--ignore-eos", action="store_true",
        help="Set ignore_eos flag when sending the benchmark request. Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code from huggingface")
    parser.add_argument("--debug", action="store_true", help="Print debug messages.")

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument("--random-input-len", type=int, default=1024, help=
        "Number of input tokens per request, used only for random sampling.")
    random_group.add_argument("--random-output-len", type=int, default=128, help=
        "Number of output tokens per request, used only for random sampling.")
    random_group.add_argument("--random-range-ratio", type=float, default=1.0, help=
                              "Range of sampled ratio of input/output length, used only for random sampling.")
    random_group.add_argument("--random-prefix-len", type=int, default=0,
        help="Number of fixed prefix tokens before random context. The length range of context in a random "
        " request is [random-prefix-len, random-prefix-len + random-prefix-len * random-range-ratio).")

@events.test_start.add_listener
def _(environment, **kw):
    print(f"Custom argument supplied: {environment.parsed_options}")

def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    prefix_token_ids = np.random.randint(0,
                                         tokenizer.vocab_size,
                                         size=prefix_len).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(prefix_token_ids +
                                  [(offsets[i] + i + j) % tokenizer.vocab_size
                                   for j in range(input_lens[i])])
        input_requests.append((prompt, int(prefix_len + input_lens[i]),
                               int(output_lens[i]), None))

    return input_requests

class MyUser(FastHttpUser):

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.args = env.parsed_options 
        self.runner = env.runner

        tokenizer_id = self.args.tokenizer if self.args.tokenizer is not None else self.args.model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=self.args.trust_remote_code)

        if self.args.dataset_name == "random":
            self.input_requests = sample_random_requests(
            prefix_len=self.args.random_prefix_len,
            input_len=self.args.random_input_len,
            output_len=self.args.random_output_len,
            num_prompts=self.args.num_prompts,
            range_ratio=self.args.random_range_ratio,
            tokenizer=tokenizer)

    @tag('openai_completion')
    @task
    def openai_completion(self):
        global prompt_id
        with lock:
            id=prompt_id
            prompt_id += 1
        #print(f'id = {id}')

        prompt, prompt_len, output_len, mm_content = self.input_requests[id%len(self.input_requests)]
        response = self.client.post(self.args.endpoint, json={
            "model": self.args.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": self.args.best_of,
            "max_tokens": output_len ,
            "logprobs": self.args.logprobs,
            "stream": True,
            "ignore_eos": self.args.ignore_eos
        })

        if self.args.debug:
            usage = response.json()['usage']
            print(f'prompt_len = {prompt_len} output_len = {output_len}')
            print(f"prompt_token = {usage['prompt_tokens']} total_tokens = {usage['total_tokens']} completion_tokens = {usage['completion_tokens']}")
            for i,out in enumerate(response.json()['choices']):
                print(f"response text [{id},{i}] = {out['text']}")

        print(f'id = {id} self.num_prompts = {self.args.num_prompts}')
        if id == (self.args.num_prompts-1):
            print(f'Number of requested prompt is reached to num_prompt {self.args.num_prompts}')
            self.runner.quit()
