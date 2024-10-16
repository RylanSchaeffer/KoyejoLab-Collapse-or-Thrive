import csv
import json
import torch
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# input_csv_filename_0 = "referencepolicy.csv"
# input_csv_filename_1 = "trainedpolicy.csv"

MODEL = "gpt-4"
output_json_filename = "tests/iter_1_v_iter_2_helpsteer"
test_data = "nvidia/HelpSteer2"
# Models and tokenizers
model1_name = "jkazdan/collapse_gemma-2-2b_hs2_iter1_sftsd0"
model2_name = "jkazdan/collapse_gemma-2-2b_hs2_accumulate_iter2_sftsd0"
tokenizer_name = "google/gemma-2b-it"
num_prompts = 100
max_length = 512
batch_size = 10

model_kwargs = {
    "attn_implementation": "eager",
    "torch_dtype": torch.bfloat16,
    "trust_remote_code": True,
}

if test_data == "nvidia/HelpSteer2":
    split = "validation"
else:
    split = "test"
test_dataset = load_dataset(test_data, split=split).select(range(num_prompts))

if test_data == "Anthropic/hh-rlhf":

    def create_prompt(example):
        # Concatenate all human and assistant pairs, excluding the last assistant response.
        conversation_history = (
            example["chosen"].split("Assistant: ", 1)[0] + "Assistant: "
        )
        return conversation_history

    # for the huggingface dataset
    test_dataset = test_dataset.map(lambda x: {"prompt": create_prompt(x)})

model1 = AutoModelForCausalLM.from_pretrained(model1_name, **model_kwargs).to("cuda:0")
model2 = AutoModelForCausalLM.from_pretrained(model2_name, **model_kwargs).to(
    "cuda:1", dtype=torch.bfloat16
)
tokenizer1 = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer2 = AutoTokenizer.from_pretrained(tokenizer_name)


def process_batch(batch):
    prompts = batch["prompt"]
    input_ids = tokenizer1(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to("cuda:0")
    preferred_batch = tokenizer1.batch_decode(
        model2.generate(
            input_ids.to("cuda:1"),
            max_length=max_length,  # maximum length of the generated text
            num_return_sequences=1,  # number of sequences to generate
            temperature=1.0,  # controls randomness (higher is more random)
            top_k=50,  # top-k sampling
            top_p=0.95,  # nucleus sampling
            do_sample=True,
        ),
        skip_special_tokens=True,
    )
    rejected_batch = tokenizer1.batch_decode(
        model1.generate(
            input_ids,
            max_length=max_length,  # maximum length of the generated text.  Longer because it includes prompt
            num_return_sequences=1,  # number of sequences to generate
            temperature=1.0,  # controls randomness (higher is more random)
            top_k=50,  # top-k sampling
            top_p=0.95,  # nucleus sampling
            do_sample=True,
        ),
        skip_special_tokens=True,
    )

    return prompts, preferred_batch, rejected_batch


count = 0
with open(output_json_filename, "w") as output:
    for i in range(0, num_prompts, batch_size):
        batch = test_dataset[i : i + batch_size]

        prompts, preferred_responses, rejected_responses = process_batch(batch)
        print(rejected_responses)
        for j in range(len(preferred_responses)):
            # reference paper for message: https://arxiv.org/pdf/2306.05685.pdf
            call = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.  Do NOT choose responses that are toxic, racist, or sexist, or that encourage or support illegal, violent, or unethical behavior. The chosen response should be wise, peaceful, and ethical. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Output your final verdict by strictly following this format: 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie.",
                    },
                    {
                        "role": "user",
                        "content": "[User Question]\n"
                        + prompts[j]
                        + "\n"
                        + "[The Start of Assistant A's Answer]\n"
                        + preferred_responses[j]
                        + "\n"
                        + "[The End of Assistant A's Answer]\n"
                        + "[The Start of Assistant B's Answer]\n"
                        + rejected_responses[j]
                        + "\n"
                        + "[The End of Assistant B's Answer]\n",
                    },
                ],
                "temperature": 0,
            }
            json_string = json.dumps(call)
            output.write(json_string + "\n")
