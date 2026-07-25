# [perf-patch mamba2a] Fast-path THUẦN DECODE cho mamba_attn._compute_common_metadata.
#
# Bằng chứng py-spy (STRATEGY.md 20/07 khuya): mỗi decode-step, builder này chạy
# ~25 torch-op CPU nhỏ; trong đó khối reclassification (prefill->decode) và
# split_decodes_and_prefills (~13 op) là control-flow THUẦN TÚY mà kết quả đã
# biết trước khi batch thuần decode: (num_reqs, 0, num_tokens, 0), không hàng
# nào cần reclassify. Fast-path bỏ đúng các op đó; MỌI tensor metadata còn lại
# tính bằng CÔNG THỨC GỐC (block indices, state_indices, cudagraph update).
# Batch có prefill / spec-decode / bất thường -> nguyên đường code gốc.
# Exact-match anchor: build FAIL nếu source drift.
import sys

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mamba_attn.py"
)

src = open(TARGET, encoding="utf-8").read()

# --- hunk 1: chèn method _decode_only_metadata ngay trước _compute_common_metadata
A1_OLD = """    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        prev_last_scheduled_idx: torch.Tensor | None = None,
    ) -> M:
        \"\"\"
        Compute metadata common to both Mamba1 and Mamba2.
        \"\"\"
        num_reqs = common_attn_metadata.num_reqs
"""
A1_NEW = """    def _decode_only_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> M:
        # [perf-patch mamba2a] Bản rút gọn của _compute_common_metadata cho
        # batch THUẦN decode (max_query_len==1, không prefilling, không spec):
        # num_prefills==0 và mọi nhánh prefill/spec đều None theo đúng logic
        # gốc; các tensor còn lại dùng NGUYÊN công thức gốc.
        num_reqs = common_attn_metadata.num_reqs
        num_decode_tokens = common_attn_metadata.num_actual_tokens

        block_idx_first_scheduled_token = None
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            state_indices_tensor = common_attn_metadata.block_table_tensor
            mamba_block_size = self.kv_cache_spec.block_size
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata, mamba_block_size
            )
        else:
            state_indices_tensor = mamba_get_block_table_tensor(
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        # torch.split([N, 0]) tương đương: d = toàn bộ (shape[0]==num_reqs,
        # bất biến mà bản gốc cũng dựa vào), p = lát rỗng cùng dtype/device.
        state_indices_tensor_d = state_indices_tensor
        state_indices_tensor_p = state_indices_tensor[num_reqs:]
        if self.vllm_config.cache_config.mamba_cache_mode != "all":
            state_indices_tensor_d = state_indices_tensor_d[
                :, : 1 + self.num_spec_tokens
            ]
            state_indices_tensor_p = state_indices_tensor_p[:, 0]

        metadata = self.metadata_cls(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=num_reqs,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=None,
            has_initial_states_p=None,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            num_accepted_tokens=None,
            query_start_loc_d=None,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=None,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_last_scheduled_token_prev_step=None,
            num_computed_tokens_p=None,
            num_reqs=num_reqs,
            seq_lens=common_attn_metadata.seq_lens,
            nums_dict=None,
            batch_ptr=None,
            token_chunk_offset_ptr=None,
        )

        return self._update_metadata_for_cudagraph_capture(metadata)

    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        prev_last_scheduled_idx: torch.Tensor | None = None,
    ) -> M:
        \"\"\"
        Compute metadata common to both Mamba1 and Mamba2.
        \"\"\"
        num_reqs = common_attn_metadata.num_reqs

        # ---- [perf-patch mamba2a] ----
        # Đường nóng: mỗi token sinh ra là một step thuần decode như vậy.
        if (
            common_attn_metadata.max_query_len == 1
            and num_accepted_tokens is None
            and prev_last_scheduled_idx is None
            and common_attn_metadata.is_prefilling is not None
            and not bool(torch.any(common_attn_metadata.is_prefilling))
        ):
            return self._decode_only_metadata(common_attn_metadata)
        # ---- [/perf-patch mamba2a] ----
"""

for name, old, new in [("decode-fastpath", A1_OLD, A1_NEW)]:
    n = src.count(old)
    if n != 1:
        sys.exit(
            "FATAL: anchor '%s' matched %d lần (cần đúng 1) — source drift, DỪNG BUILD."
            % (name, n)
        )
    src = src.replace(old, new, 1)

open(TARGET, "w", encoding="utf-8").write(src)
print("patch_mamba2a: applied 1/1 hunk OK ->", TARGET)
