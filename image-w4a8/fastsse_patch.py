"""[perf-patch fastsse] Fast-path SSE cho chat streaming (chay luc BUILD).

VI SAO: portal ep entrypoint `python3 -m vllm.entrypoints.openai.api_server`, ma
`api_server.py run_server()` goi `run_server_worker()` DUNG 1 LAN (khong fork) va binary Rust
frontend khong co trong image => frontend la MOT process Python vua chay event loop, vua render
chat-template + tokenize ~4400 tok/req, vua serialize ~8000 SSE frame/s cho ~27 stream tren 3 core.
py-spy tren chinh stack nay: `pydantic model_dump_json` = leaf lon nhat (47 samples).
Patch nay bo han dung + serialize pydantic moi token, thay bang string template trong _fastsse.py.

OUTPUT-PRESERVING: fast-path CHI bat khi khong co tool/reasoning/logprobs/usage/logging va chi cho
delta content thuan giua stream (finish_reason is None). Moi case khac di NGUYEN duong code goc.
Frame byte-identical voi model_dump_json(exclude_unset=True) — ep tai build time boi verify_fastsse.py.
KHONG gop chunk, KHONG doi thoi diem phat token (khac han --stream-interval, xem STRATEGY.md muc 6).

Tai lap image online-r6 = online-r5 + script nay. Reference diff: fastsse_serving.patch.
"""

from pathlib import Path

SERVING = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
    "chat_completion/serving.py"
)
MARKER = "# [perf-patch fastsse]"

src = SERVING.read_text(encoding="utf-8")
assert MARKER not in src, "already patched"

# --- 1) import module template -------------------------------------------------
A1 = "import io\nimport json\nimport time\n"
assert src.count(A1) == 1, "anchor import khong duy nhat"
src = src.replace(
    A1,
    "import io\nimport json\n\n"
    "from vllm.entrypoints.openai.chat_completion import _fastsse  " + MARKER + "\n"
    "import time\n",
    1,
)

# --- 2) quyet dinh bat fast-path 1 lan cho ca stream ---------------------------
A2 = (
    "            stream_options, self.enable_force_include_usage\n"
    "        )\n\n"
    "        try:\n"
)
assert src.count(A2) == 1, "anchor quyet dinh fast-path khong duy nhat"
src = src.replace(
    A2,
    "            stream_options, self.enable_force_include_usage\n"
    "        )\n\n"
    "        # ---- [perf-patch fastsse] ------------------------------------------\n"
    "        # Fast-path serializer cho case streaming phổ thông (delta content\n"
    "        # thuần). Frame byte-identical với model_dump_json(exclude_unset=True)\n"
    "        # — được assert lúc build image (verify_fastsse.py). Mọi request có\n"
    "        # tool/reasoning/logprobs/usage/logging giữ NGUYÊN đường code gốc.\n"
    "        _fp_head = None\n"
    "        if (\n"
    "            not self.use_harmony\n"
    "            and self.parser_cls is None\n"
    "            and reasoning_parser is None\n"
    "            and not is_mistral_grammar_path\n"
    "            and not tool_choice_auto\n"
    "            and tool_choice_function_name is None\n"
    '            and request.tool_choice != "required"\n'
    "            and not (request.logprobs and request.top_logprobs is not None)\n"
    "            and not request.return_token_ids\n"
    "            and not include_continuous_usage\n"
    "            and not self.enable_log_outputs\n"
    "        ):\n"
    "            _fp_head = _fastsse.build_head(request_id, created_time, model_name)\n"
    "        # ---- [/perf-patch fastsse] -----------------------------------------\n\n"
    "        try:\n",
    1,
)

# --- 3) phat frame fast-path -----------------------------------------------------
A3 = (
    "                    if output.finish_reason is None:\n"
    "                        # Send token-by-token response for each request.n\n"
)
assert src.count(A3) == 1, "anchor yield frame khong duy nhat"
src = src.replace(
    A3,
    "                    # ---- [perf-patch fastsse] ----\n"
    "                    if (\n"
    "                        _fp_head is not None\n"
    "                        and output.finish_reason is None\n"
    "                        and delta_message.content is not None\n"
    "                    ):\n"
    "                        yield _fastsse.frame(_fp_head, i, delta_message.content)\n"
    "                        continue\n"
    "                    # ---- [/perf-patch fastsse] ----\n" + A3,
    1,
)

SERVING.write_text(src, encoding="utf-8")

out = SERVING.read_text(encoding="utf-8")
assert out.count("import _fastsse  " + MARKER) == 1, "hunk 1 (import) missing"
assert out.count("_fp_head = _fastsse.build_head(") == 1, "hunk 2 (gate) missing"
assert out.count("yield _fastsse.frame(_fp_head, i, delta_message.content)") == 1, (
    "hunk 3 (frame) missing"
)
compile(out, str(SERVING), "exec")
print("fastsse patch applied OK (3 hunks)")
