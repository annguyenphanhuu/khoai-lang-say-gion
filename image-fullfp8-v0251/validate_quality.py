"""Sanity-check the incremental ShortConv + LM-head FP8 weight error in HF."""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "/model"
PROMPTS = [
    "Compute 37 * 19. Return only the integer.",
    "What is the capital city of France? Answer in one short sentence.",
    "Which planet is the largest in our Solar System? Answer briefly.",
    "A train travels 120 km in 2 hours. What is its average speed in km/h?",
    "Translate 'good morning' into Vietnamese. Return only the translation.",
]


def qdq_fp8(weight: torch.Tensor) -> torch.Tensor:
    """Approximate vLLM online per-tensor E4M3 weight quantization."""
    max_abs = weight.abs().max().float()
    scale = (max_abs / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    return (
        (weight.float() / scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .to(torch.bfloat16)
        * scale
    )


tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda")
model.eval()


def generate_all() -> list[str]:
    answers = []
    for prompt in PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=48,
                use_cache=True,
            )
        answers.append(
            tokenizer.decode(
                output[0, encoded.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
        )
    return answers


baseline = generate_all()

quantized = []
with torch.no_grad():
    for name, parameter in model.named_parameters():
        if name.endswith(("conv.in_proj.weight", "conv.out_proj.weight")):
            parameter.copy_(qdq_fp8(parameter))
            quantized.append(name)

    # Keep input embeddings BF16. The vLLM patch similarly breaks the tied
    # Parameter only when the LM head is converted to its online FP8 copy.
    lm_head = model.get_output_embeddings()
    input_embedding = model.get_input_embeddings()
    assert lm_head is not None
    assert lm_head.weight.data_ptr() == input_embedding.weight.data_ptr()
    lm_head.weight = torch.nn.Parameter(
        qdq_fp8(lm_head.weight.detach()), requires_grad=False
    )
    assert lm_head.weight.data_ptr() != input_embedding.weight.data_ptr()
    quantized.append("lm_head.weight (untied FP8 simulation)")

candidate = generate_all()
print(json.dumps({"quantized": quantized, "baseline": baseline, "candidate": candidate}, ensure_ascii=False, indent=2))
