from typing import Dict

def format_prompt(format_dict: Dict, sys_name="alpaca"):
    if sys_name == "alpaca":
        prompt_dict = {
            "prompt_input": (
                "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
                "### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:\n"
            ),
            "prompt_no_input": (
                "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n"
                "### 指示:\n{instruction}\n\n### 応答:\n"
            ),
        }
        if "input" not in format_dict or format_dict["input"] is None or format_dict["input"] == "" or format_dict["input"].isspace():
            return prompt_dict['prompt_no_input'].format_map(format_dict)
        else:
            return prompt_dict["prompt_input"].format_map(format_dict)

    elif sys_name == "shortqa":
        prompt =  (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request using a single word or phrase.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
        return prompt.format_map(format_dict)

    elif sys_name == "qg":  # question_generation
        prompt = (
            "Generate a question whose answer is:\n{instruction}\n\n"
            "Question:\n"
        )
        return prompt.format_map(format_dict)

    elif sys_name == "caption":
        return ""

    elif sys_name == "None":
        return "{instruction}".format_map(format_dict)

    else:
        ValueError(sys_name)
