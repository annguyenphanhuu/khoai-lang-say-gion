# [perf-patch fastsse] — NGUỒN SỰ THẬT DUY NHẤT cho template SSE fast-path.
# Sinh frame byte-identical với ChatCompletionStreamResponse.model_dump_json(
# exclude_unset=True) cho case phổ thông: 1 delta content thuần, không tool/
# reasoning/logprobs/usage. Tính đúng đắn được ÉP tại build time bởi
# verify_fastsse.py (build fail nếu lệch 1 byte). Chỉ tối ưu hiệu năng CPU —
# không đổi behavior, không đổi output; mọi case khác đi đường code gốc.
import json

MID = ',"delta":{"content":'
TAIL = '},"logprobs":null,"finish_reason":null,"token_ids":null}]}\n\n'


def build_head(request_id: str, created: int, model: str) -> str:
    return 'data: {"id":%s,"object":"chat.completion.chunk","created":%d,"model":%s,"choices":[{"index":' % (
        json.dumps(request_id, ensure_ascii=False),
        created,
        json.dumps(model, ensure_ascii=False),
    )


def frame(head: str, index: int, content: str) -> str:
    return "%s%d%s%s%s" % (
        head,
        index,
        MID,
        json.dumps(content, ensure_ascii=False),
        TAIL,
    )
