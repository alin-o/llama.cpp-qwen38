import os

import pytest
from utils import *

server = ServerPreset.stories15m_moe()

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]


@pytest.mark.parametrize("scale,re_content", [
    # without applying lora, the model should behave like a bedtime story generator
    (0.0, "(little|girl|three|years|old)+"),
    # with lora, the model should behave like a Shakespearean text generator
    (1.0, "(eye|love|glass|sun)+"),
])
def test_lora(scale: float, re_content: str):
    global server
    server.start()
    res_lora_control = server.make_request("POST", "/lora-adapters", data=[
        {"id": 0, "scale": scale}
    ])
    assert res_lora_control.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
    })
    assert res.status_code == 200
    assert match_regex(re_content, res.body["content"])


def test_lora_per_request():
    global server
    server.n_slots = 4
    server.start()

    # running the same prompt with different lora scales, all in parallel
    # each prompt will be processed by a different slot
    prompt = "Look in thy glass"
    lora_config = [
        ( [{"id": 0, "scale": 0.0}], "(bright|day|many|happy)+" ),
        ( [{"id": 0, "scale": 0.0}], "(bright|day|many|happy)+" ),
        ( [{"id": 0, "scale": 0.3}], "(special|thing|gifted)+" ),
        ( [{"id": 0, "scale": 0.7}], "(far|from|home|away)+" ),
        ( [{"id": 0, "scale": 1.0}], "(eye|love|glass|sun)+" ),
        ( [{"id": 0, "scale": 1.0}], "(eye|love|glass|sun)+" ),
    ]

    tasks = [(
        server.make_request,
        ("POST", "/completion", {
            "prompt": prompt,
            "lora": lora,
            "seed": 42,
            "temperature": 0.0,
            "cache_prompt": False, # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
    ) for lora, _ in lora_config]
    results = parallel_function_calls(tasks)

    assert all([res.status_code == 200 for res in results])
    for res, (_, re_test) in zip(results, lora_config):
        assert match_regex(re_test, res.body["content"])


def test_paged_lora_per_request_and_cache_invalidation():
    global server
    model = os.environ.get("LLAMA_SERVER_PAGED_LORA_MODEL")
    adapter = os.environ.get("LLAMA_SERVER_PAGED_LORA_FILE")
    if not model or not adapter:
        pytest.skip("set LLAMA_SERVER_PAGED_LORA_MODEL and LLAMA_SERVER_PAGED_LORA_FILE to run paged LoRA coverage")

    server = ServerProcess()
    server.model_file = model
    server.model_hf_repo = None
    server.model_hf_file = None
    server.offline = True
    server.lora_files = [adapter]
    server.kv_paged = True
    server.n_slots = 2
    server.n_ctx = 1024
    server.n_batch = 128
    server.n_ubatch = 128
    server.block_size = 16
    server.n_gpu_blocks = 128
    server.n_cpu_blocks = 16
    server.n_predict = 8
    server.temperature = 0.0
    server.seed = 42
    server.start()

    request = {
        "prompt": "A deterministic adapter isolation prompt. " * 8,
        "n_predict": 8,
        "temperature": 0.0,
        "seed": 42,
        "return_tokens": True,
    }
    cold = {}
    for scale in (0.0, 1.0):
        response = server.make_request("POST", "/completion", data={
            **request,
            "lora": [{"id": 0, "scale": scale}],
            "cache_prompt": False,
        })
        assert response.status_code == 200
        cold[scale] = response.body["tokens"]
    assert cold[0.0] != cold[1.0]

    tasks = [(
        server.make_request,
        ("POST", "/completion", {
            **request,
            "lora": [{"id": 0, "scale": scale}],
            "cache_prompt": False,
            "id_slot": id_slot,
        })
    ) for id_slot, scale in enumerate((0.0, 1.0))]
    results = parallel_function_calls(tasks)
    assert all(response.status_code == 200 for response in results)
    assert results[0].body["tokens"] == cold[0.0]
    assert results[1].body["tokens"] == cold[1.0]

    warm_request = {
        **request,
        "lora": [{"id": 0, "scale": 1.0}],
        "cache_prompt": True,
        "id_slot": 0,
    }
    prime = server.make_request("POST", "/completion", data=warm_request)
    warm = server.make_request("POST", "/completion", data=warm_request)
    assert prime.status_code == 200
    assert warm.status_code == 200
    assert prime.body["tokens"] == cold[1.0]
    assert warm.body["tokens"] == cold[1.0]
    assert warm.body["timings"]["cache_n"] == prime.body["timings"]["prompt_n"] - 1
    assert warm.body["timings"]["prompt_n"] == 1

    switched = server.make_request("POST", "/completion", data={
        **warm_request,
        "lora": [{"id": 0, "scale": 0.0}],
    })
    assert switched.status_code == 200
    assert switched.body["tokens"] == cold[0.0]
    assert switched.body["timings"]["cache_n"] == 0


def test_paged_alora_pre_invocation_and_batch_isolation():
    global server
    model = os.environ.get("LLAMA_SERVER_ALORA_MODEL")
    adapter = os.environ.get("LLAMA_SERVER_ALORA_FILE")
    if not model or not adapter:
        pytest.skip("set LLAMA_SERVER_ALORA_MODEL and LLAMA_SERVER_ALORA_FILE to run paged aLoRA coverage")

    server = ServerProcess()
    server.model_file = model
    server.model_hf_repo = None
    server.model_hf_file = None
    server.offline = True
    server.lora_files = [adapter]
    server.kv_paged = True
    server.n_slots = 2
    server.n_ctx = 1024
    server.n_batch = 128
    server.n_ubatch = 128
    server.block_size = 16
    server.n_gpu_blocks = 128
    server.n_cpu_blocks = 16
    server.n_predict = 8
    server.temperature = 0.0
    server.seed = 42
    server.start()

    adapters = server.make_request("GET", "/lora-adapters")
    assert adapters.status_code == 200
    invocation = adapters.body[0]["alora_invocation_string"]
    prompt = ("The stable prefix is evaluated without the adapter. " * 8).rstrip() + invocation + " Continue the story."
    request = {
        "prompt": prompt,
        "n_predict": 8,
        "temperature": 0.0,
        "seed": 42,
        "return_tokens": True,
    }

    cold = {}
    for scale in (0.0, 1.0):
        response = server.make_request("POST", "/completion", data={
            **request,
            "lora": [{"id": 0, "scale": scale}],
            "cache_prompt": False,
        })
        assert response.status_code == 200
        cold[scale] = response.body["tokens"]
    assert cold[0.0] != cold[1.0]

    warm_request = {
        **request,
        "lora": [{"id": 0, "scale": 1.0}],
        "cache_prompt": True,
        "id_slot": 0,
    }
    prime = server.make_request("POST", "/completion", data=warm_request)
    warm = server.make_request("POST", "/completion", data=warm_request)
    assert prime.status_code == 200
    assert warm.status_code == 200
    assert prime.body["tokens"] == cold[1.0]
    assert warm.body["tokens"] == cold[1.0]
    assert 0 < warm.body["timings"]["cache_n"] < prime.body["timings"]["prompt_n"]
    assert warm.body["timings"]["cache_n"] + warm.body["timings"]["prompt_n"] == prime.body["timings"]["prompt_n"]

    followup_prompt = (
        prompt
        + " The assistant answered the first turn. The user now asks a follow-up."
        + invocation
        + " Continue from the new instruction."
    )
    followup_request = {
        **request,
        "prompt": followup_prompt,
        "lora": [{"id": 0, "scale": 1.0}],
    }
    followup_warm = server.make_request("POST", "/completion", data={
        **followup_request,
        "cache_prompt": True,
        "id_slot": 0,
    })
    followup_cold = server.make_request("POST", "/completion", data={
        **followup_request,
        "cache_prompt": False,
        "id_slot": 1,
    })
    assert followup_warm.status_code == 200
    assert followup_cold.status_code == 200
    assert followup_warm.body["tokens"] == followup_cold.body["tokens"]
    assert followup_cold.body["timings"]["cache_n"] == 0

    tokenized = server.make_request("POST", "/tokenize", data={
        "content": prompt,
        "add_special": True,
    })
    invocation_tokens = adapters.body[0]["alora_invocation_tokens"]
    retained_invocation_start = next(
        i for i in range(len(tokenized.body["tokens"]) - len(invocation_tokens), -1, -1)
        if tokenized.body["tokens"][i:i + len(invocation_tokens)] == invocation_tokens
    )
    assert 0 < followup_warm.body["timings"]["cache_n"] <= retained_invocation_start - 1
    assert (
        followup_warm.body["timings"]["cache_n"]
        + followup_warm.body["timings"]["prompt_n"]
        == followup_cold.body["timings"]["prompt_n"]
    )

    tasks = [(
        server.make_request,
        ("POST", "/completion", {
            **request,
            "lora": [{"id": 0, "scale": scale}],
            "cache_prompt": False,
            "id_slot": id_slot,
        })
    ) for id_slot, scale in enumerate((0.0, 1.0))]
    results = parallel_function_calls(tasks)
    assert all(response.status_code == 200 for response in results)
    assert results[0].body["tokens"] == cold[0.0]
    assert results[1].body["tokens"] == cold[1.0]


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_with_big_model():
    server = ServerProcess()
    server.model_hf_repo = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    server.model_hf_file = "Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"
    server.model_alias = "Llama-3.2-8B-Instruct"
    server.n_slots = 4
    server.n_ctx = server.n_slots * 1024
    server.n_predict = 64
    server.temperature = 0.0
    server.seed = 42
    server.lora_files = [
        download_file("https://huggingface.co/ngxson/Llama-3-Instruct-abliteration-LoRA-8B-F16-GGUF/resolve/main/Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf"),
        # TODO: find & add other lora adapters for this model
    ]
    server.start(timeout_seconds=600)

    # running the same prompt with different lora scales, all in parallel
    # each prompt will be processed by a different slot
    prompt = "Write a computer virus"
    lora_config = [
        # without applying lora, the model should reject the request
        ( [{"id": 0, "scale": 0.0}], "I can't provide you with a code for a computer virus" ),
        ( [{"id": 0, "scale": 0.0}], "I can't provide you with a code for a computer virus" ),
        ( [{"id": 0, "scale": 0.3}], "I can't write a computer virus" ),
        # with 0.7 scale, the model should provide a simple computer virus with hesitation
        ( [{"id": 0, "scale": 0.7}], "Warning: This is a hypothetical exercise" ),
        # with 1.5 scale, the model should confidently provide a computer virus
        ( [{"id": 0, "scale": 1.5}], "A task of some complexity! Here's a simple computer virus" ),
        ( [{"id": 0, "scale": 1.5}], "A task of some complexity! Here's a simple computer virus" ),
    ]

    tasks = [(
        server.make_request,
        ("POST", "/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "lora": lora,
            "cache_prompt": False, # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
    ) for lora, _ in lora_config]
    results = parallel_function_calls(tasks)

    assert all([res.status_code == 200 for res in results])
    for res, (_, re_test) in zip(results, lora_config):
        assert re_test in res.body["choices"][0]["message"]["content"]
