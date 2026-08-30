import pytest
from utils import *

# We use a F16 MOE gguf as main model, and q4_0 as draft model

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # set default values
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


def test_with_and_without_draft_split_batch():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "seed": 4242,
        "n_predict": 16,
        "return_tokens": True,
    }

    server.model_draft = None
    server.spec_type = None
    server.n_batch = 2
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()

    create_server()
    server.n_batch = 2
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    tokens_draft = res.body["tokens"]

    assert tokens_no_draft == tokens_draft


def test_paged_mtp_matches_target_greedy(monkeypatch):
    global server
    model = os.environ.get("LLAMA_SERVER_PAGED_MODEL")
    model_draft = os.environ.get("LLAMA_SERVER_MTP_MODEL")
    if not model or not model_draft:
        pytest.skip("set LLAMA_SERVER_PAGED_MODEL and LLAMA_SERVER_MTP_MODEL to run paged MTP equivalence")

    # This obsolete diagnostic setting must not change verification semantics.
    monkeypatch.setenv("LLAMA_SPEC_DIAG_LEGACY_BATCH", "1")

    prompts = [
        "Explain to a curious high-school student why the sky is blue during the day but often red or orange near sunset. Include the roles of wavelength, scattering, and the longer path through the atmosphere.",
        "A water tank is initially 30 percent full. A pump adds 18 liters per minute while a leak removes 3 liters per minute. After 14 minutes the tank is 65 percent full. Find the tank's total capacity and show the calculation step by step.",
    ]
    request = {
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1234,
        "n_predict": 128,
        "ignore_eos": True,
        "cache_prompt": False,
        "return_tokens": True,
    }

    def configure(model_draft_path):
        global server
        server = ServerPreset.stories15m_moe()
        server.model_file = model
        server.model_hf_repo = None
        server.model_hf_file = None
        server.model_draft = model_draft_path
        server.spec_type = "draft-mtp" if model_draft_path else None
        server.spec_draft_n_min = 1
        server.spec_draft_n_max = int(os.environ.get("LLAMA_SERVER_MTP_N_MAX", "3"))
        server.n_gpu_layer = 99
        server.n_batch = 1024
        server.n_ubatch = 1024
        server.n_slots = 1
        server.server_port = 18089
        server.ctk = os.environ.get("LLAMA_SERVER_MTP_KV_TYPE", "q8_0")
        server.ctv = server.ctk
        server.fa = "on"
        server.kv_paged = True

    def generate_all():
        responses = [
            server.make_request("POST", "/completion", data={**request, "prompt": prompt})
            for prompt in prompts
        ]
        for res in responses:
            assert res.status_code == 200
            assert len(res.body["tokens"]) == request["n_predict"]
        return responses

    configure(None)
    server.start(timeout_seconds=180)
    expected = [res.body["tokens"] for res in generate_all()]
    server.stop()

    configure(model_draft)
    server.start(timeout_seconds=180)
    responses = generate_all()
    for res, tokens_target in zip(responses, expected):
        assert res.body["timings"]["draft_n"] > 0
        assert res.body["tokens"] == tokens_target


def test_different_draft_min_draft_max():
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_slot_ctx_not_exceeded():
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])
