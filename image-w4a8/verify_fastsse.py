# Build-time gate: fast-path frame PHẢI byte-identical với pydantic
# model_dump_json(exclude_unset=True) trên bộ case hiểm — lệch 1 byte = FAIL.
import sys

from vllm.entrypoints.openai.chat_completion import _fastsse
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)

CONTENTS = [
    "",
    " ",
    "hello world",
    'quote " backslash \\ slash / end',
    "newline\ntab\tcr\r",
    "unicode: Hé ✓ 日本語 tiếng Việt đắt 🚀🚀",
    "control:\x00\x01\x1f\x7f",
    "line sep   para sep  ",
    "<|im_end|></s>{}[]:,",
    "long " + "x" * 4096,
    "mixed é́ combining",
]
IDS = ["chatcmpl-abc123", "chatcmpl-ựứ-weird\"id"]
MODELS = ["LFM2.5-1.2B-Instruct", 'mo"del\\name']
INDEXES = [0, 1, 7]

n = 0
for rid in IDS:
    for model in MODELS:
        for i in INDEXES:
            head = _fastsse.build_head(rid, 1753000000, model)
            for content in CONTENTS:
                choice = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(content=content),
                    logprobs=None,
                    finish_reason=None,
                    token_ids=None,
                )
                chunk = ChatCompletionStreamResponse(
                    id=rid,
                    object="chat.completion.chunk",
                    created=1753000000,
                    choices=[choice],
                    model=model,
                )
                want = "data: %s\n\n" % chunk.model_dump_json(exclude_unset=True)
                got = _fastsse.frame(head, i, content)
                if got != want:
                    sys.exit(
                        "FATAL byte mismatch:\n got=%r\nwant=%r" % (got, want)
                    )
                n += 1

# patched serving.py phải import sạch và chứa marker
import vllm.entrypoints.openai.chat_completion.serving as serving_mod

src = open(serving_mod.__file__, encoding="utf-8").read()
assert src.count("[perf-patch fastsse]") >= 3, "patch markers missing"
print("verify_fastsse OK: %d cases byte-identical; serving.py imports clean" % n)
