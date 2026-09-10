# Qwen3.8-27B model-only replacement

Author request (2026-09-10): save the running Qwen3.6 results, then use
Qwen3.8-27B with all other latest-version requirements unchanged.

The source contract is
`../secopd_qwen36_roles_think_20260910/EXECUTION_CONTRACT.md`.
This is generation authorization, not an optimizer launch or new paid labeling.

## Identity and invariant inputs

- Replace Qwen/Qwen3.6-27B with official Qwen/Qwen3.8-27B at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Use the identical September 10 source rows, row order, 9,600 task/injection
  pairs, attack positions, train/dev assignments and references. Do not redraw
  positions or substitute any saved response from another model.
- Retokenize the same Student/T-minus and exact-q1 T-plus rendered text with
  the replacement's own tokenizer. Record whether native token IDs changed.
- Preserve user/input roles, thinking prefix, temperature=1, top_p=1,
  top_k=-1, min_p=0, neutral penalties, per-source seeds, max_new_tokens=16,384,
  total context=32,768, two TP=2 instances on GPU 2/5 and GPU 3/4, BF16,
  eight concurrent requests per instance, and the pinned vLLM 0.24.0 image.
- Keep the existing manual text renderer; do not add a lower reasoning-effort
  directive, enable speculative decoding, quantize weights, or adopt model-card
  sampling defaults as a side effect of the model replacement.
- Keep T-plus privilege, attention quarantine, all objectives and reliability
  code unchanged. This generation run does not validate a 27B T-plus forward.

## Save, validate and launch

1. The user-authorized SIGUSR1 stop request is recorded in the Qwen3.6 run as
   `USER_MODEL_SWITCH_REQUEST_20260910.json`. Its existing supervisor stops
   new submissions, drains pending calls and releases only its own servers.
   Preserve raw requests, responses, exact trajectories, journals and hashes.
2. Verify every cached Qwen3.8 asset against the official pinned repository
   manifest: LFS SHA256 or Git blob SHA1, recording SHA256 for every file.
   A separate read-only model staging directory uses hardlinks to verified
   cache blobs; do not alter the shared cache or other model jobs.
3. Run the existing source/rendering/token/stop/quarantine CPU regressions.
   Prove that the worker, server, final assembly and supervisor functions are
   unchanged, except the new executable filename and model/run constants.
4. The staging gate checks all 9,600 requests against the prior run. Only model
   identity/passport metadata and native prompt token IDs may differ. Source
   rows, text, seed and all non-model sampling parameters must remain identical.
5. Use the exact same 12 canary source indices (six per lane, two per attack
   cell). Keep the prior engineering gate: verified exact tokens, one thinking
   end, nonempty final answer and natural EOS. Count them toward 9,600.
6. Once both canaries pass, automatically launch the two disjoint full shards.
   Fresh Qwen3.8 responses are collected for all 9,600 sources. No Qwen3.6
   responses or A/U labels count toward the new model's output.
7. Preserve the previous 3,600-second request timeout, 48-hour run limit,
   error/drain behavior, complete-coverage audit and automatic owned-GPU release.

Existing temperature-zero API annotation remains a separate active task.
The balance-held API group stays held. No other GPU process is stopped.

## Evidence and speed boundary

The official checkpoint identity and architecture were checked using:
https://huggingface.co/Qwen/Qwen3.8-27B
and the pinned Hugging Face model-info response stored under
`artifacts/qwen38_model_switch_preflight_20260910/official_model_info.json`.
The official model card's recommended top_p/top_k are not adopted because the
user requested the same current experimental settings.

Architecture remains Qwen3_5ForConditionalGeneration with hybrid linear/full
attention. Model replacement alone does not establish a throughput benefit.
Report observed token counts and wall time, without promising a speed-up.
After generation, new A/U judgment, capped-output disposition, training input
integration, true T-plus forwards and model-specific reliability calibration
remain separate requirements. Generation is not training or scientific evidence.

Run directory:
`artifacts/secopd_qwen38_27b_9600_roles_think_t1_16k_20260910`.

The existing four-line progress view is copied unchanged into the new run.
Read `model_only_diff.json`, `prompt_gate.json`, `model_verification.json`,
`manifest.json`, and then the live `progress.json` and container receipts.
