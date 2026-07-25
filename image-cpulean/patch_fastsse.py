# Áp patch fastsse vào serving.py của vLLM trong image, theo kiểu exact-match:
# nếu anchor không tồn tại hoặc xuất hiện !=1 lần (source drift) -> FAIL BUILD.
import sys

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
    "chat_completion/serving.py"
)

src = open(TARGET, encoding="utf-8").read()

# --- patch 1: import module template ở đầu file -----------------------------
A1_OLD = "import json\n"
A1_NEW = (
    "import json\n"
    "\n"
    "from vllm.entrypoints.openai.chat_completion import _fastsse  # [perf-patch fastsse]\n"
)

# --- patch 2: setup per-request (sau khi include_usage được tính) ------------
A2_OLD = """        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )
"""
A2_NEW = A2_OLD + """
        # ---- [perf-patch fastsse] ------------------------------------------
        # Fast-path serializer cho case streaming phổ thông (delta content
        # thuần). Frame byte-identical với model_dump_json(exclude_unset=True)
        # — được assert lúc build image (verify_fastsse.py). Mọi request có
        # tool/reasoning/logprobs/usage/logging giữ NGUYÊN đường code gốc.
        _fp_head = None
        if (
            not self.use_harmony
            and self.parser_cls is None
            and reasoning_parser is None
            and not is_mistral_grammar_path
            and not tool_choice_auto
            and tool_choice_function_name is None
            and request.tool_choice != "required"
            and not (request.logprobs and request.top_logprobs is not None)
            and not request.return_token_ids
            and not include_continuous_usage
            and not self.enable_log_outputs
        ):
            _fp_head = _fastsse.build_head(request_id, created_time, model_name)
        # ---- [/perf-patch fastsse] -----------------------------------------
"""

# --- patch 3: fast-path per-token (trước nhánh finish_reason) ----------------
A3_OLD = """                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
"""
A3_NEW = """                    # ---- [perf-patch fastsse] ----
                    if (
                        _fp_head is not None
                        and output.finish_reason is None
                        and delta_message.content is not None
                    ):
                        yield _fastsse.frame(_fp_head, i, delta_message.content)
                        continue
                    # ---- [/perf-patch fastsse] ----
                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
"""

for name, old, new in [("import", A1_OLD, A1_NEW), ("setup", A2_OLD, A2_NEW), ("hotloop", A3_OLD, A3_NEW)]:
    n = src.count(old)
    if n != 1:
        sys.exit("FATAL: anchor '%s' matched %d lần (cần đúng 1) — source drift, DỪNG BUILD." % (name, n))
    src = src.replace(old, new, 1)

open(TARGET, "w", encoding="utf-8").write(src)
print("patch_fastsse: applied 3/3 hunks OK ->", TARGET)
