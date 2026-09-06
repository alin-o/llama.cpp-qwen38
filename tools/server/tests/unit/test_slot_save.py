import pytest
from utils import *
import base64
import io
import math
import re
import requests
import struct
import wave
from concurrent.futures import ThreadPoolExecutor

# sequence state file: magic(4) version(4) payload_size(4), then payload_size llama_token words
STATE_FILE_HEADER_SIZE = 12

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = str(tmp_path)
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_legacy_token_list():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # rewrite the token payload into a plain token list, as written by servers that predate the packed server_tokens format
    path = os.path.join(server.slot_save_path, "slot_legacy.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    # the payload written by this server starts with a packed header: LLAMA_TOKEN_NULL(4) version(4) n_tokens(4)
    packed_header_size = 12

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    assert n_tokens == 84

    tokens_start = STATE_FILE_HEADER_SIZE + packed_header_size
    data = data[:STATE_FILE_HEADER_SIZE] + data[tokens_start:tokens_start + n_tokens * 4] + data[payload_end:]
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, n_tokens)

    with open(path, "wb") as f:
        f.write(data)

    # the plain token list must restore, and the restored KV must be reusable
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 6  # only the different part is processed



def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


#
# Multimodal server (mmproj loaded) slot save/restore.
#
# A pure-text slot on a multimodal server and a slot containing images must both support save/restore.
# Erase remains gated on the slot's content.
#

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"
IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


def _get_wav_base64(frequency: float) -> str:
    sample_rate = 16000
    samples = [
        int(12000 * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(sample_rate)
    ]
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return base64.b64encode(output.getvalue()).decode("utf-8")


@pytest.fixture
def mmproj_server():
    # tinygemma3 is a small multimodal model: the mmproj is provided by the HF registry API and auto-downloaded on first run.
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    mm_server = ServerPreset.tinygemma3()
    if os.environ.get("LLAMA_TEST_MODEL") and os.environ.get("LLAMA_TEST_MMPROJ"):
        mm_server.model_hf_repo = None
        mm_server.model_file = os.environ["LLAMA_TEST_MODEL"]
        mm_server.mmproj_file = os.environ["LLAMA_TEST_MMPROJ"]
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
    return mm_server


@pytest.fixture
def audio_mmproj_server():
    model = os.environ.get("LLAMA_TEST_AUDIO_MODEL", "/models/LFM2-Audio-1.5B-Q8_0.gguf")
    mmproj = os.environ.get("LLAMA_TEST_AUDIO_MMPROJ", "/models/mmproj-LFM2-Audio-1.5B-Q8_0.gguf")
    if not os.path.isfile(model) or not os.path.isfile(mmproj):
        pytest.skip("audio checkpoint test requires LLAMA_TEST_AUDIO_MODEL and LLAMA_TEST_AUDIO_MMPROJ")

    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    audio_server = ServerProcess()
    audio_server.offline = True
    audio_server.model_hf_repo = None
    audio_server.model_hf_file = None
    audio_server.model_file = model
    audio_server.mmproj_file = mmproj
    audio_server.model_alias = "audio-checkpoint-test"
    audio_server.n_gpu_layer = 1
    audio_server.slot_save_path = "./tmp"
    audio_server.temperature = 0.0
    return audio_server


def test_slot_save_restore_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    # A pure-text prompt processed on slot 1 of a multimodal server.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Saving a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0  # the slot KV (prompt + generated tokens) was written

    # Restore the saved state into slot 0; it must round-trip exactly.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # Prefix reuse is not checked with the default SWA cache.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_global_checkpoint_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.n_ctx = 2048
    server.n_batch = 64
    server.n_ubatch = 32
    server.kv_unified = True
    server.server_continuous_batching = True
    server.ctx_checkpoints = 8
    server.checkpoint_min_step = 8
    prefix = "A knight crossed the quiet valley before sunrise. " * 100
    followup_prompt = prefix + "Then the bells rang from the northern tower."

    server.start()
    first = server.make_request("POST", "/completion", data={
        "prompt": prefix,
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0,
    })
    assert first.status_code == 200

    followup = server.make_request("POST", "/completion", data={
        "prompt": followup_prompt,
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0,
    })
    assert followup.status_code == 200
    checkpoint = followup.body["timings"]["checkpoint"]
    assert checkpoint["hit_tokens"] > 0
    assert checkpoint["target_cells"] > 0
    assert checkpoint["hidden_replay"] == 0


def test_global_checkpoint_text_restore_then_image(mmproj_server):
    server = mmproj_server
    server.n_slots = 4
    server.n_ctx = 4096
    server.n_batch = 512
    server.n_ubatch = 512
    server.n_predict = 4
    server.kv_unified = True
    server.server_continuous_batching = True
    server.server_metrics = True
    server.ctx_checkpoints = 8
    server.checkpoint_min_step = 8
    prefix = "A knight crossed the quiet valley before sunrise. " * 100
    image_cat = _get_img_base64(IMG_URL_CAT)
    image_truck = _get_img_base64(IMG_URL_TRUCK)

    server.start()
    first = server.make_request("POST", "/completion", data={
        "prompt": prefix,
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0,
    })
    assert first.status_code == 200

    restored = server.make_request("POST", "/completion", data={
        "prompt": prefix + "Then the bells rang from the northern tower.",
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0,
    })
    assert restored.status_code == 200
    assert restored.body["timings"]["checkpoint"]["hit_tokens"] > 0

    image = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [image_cat],
        },
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert image.status_code == 200
    assert image.body["timings"]["prompt_n"] > 32

    media_prefix = "Describe this image: <__media__>\n" + "Give a careful visual answer. " * 20
    suffix = "Finally, answer briefly."

    oracle = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": media_prefix + suffix,
            "multimodal_data": [image_cat],
        },
        "id_slot": 2,
        "cache_prompt": False,
        "temperature": 0,
        "top_k": 1,
        "return_tokens": True,
    })
    assert oracle.status_code == 200

    cold = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": media_prefix,
            "multimodal_data": [image_cat],
        },
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert cold.status_code == 200

    warm = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": media_prefix + suffix,
            "multimodal_data": [image_cat],
        },
        "id_slot": 3,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
        "return_tokens": True,
    })
    assert warm.status_code == 200
    assert warm.body["tokens"] == oracle.body["tokens"]
    checkpoint = warm.body["timings"]["checkpoint"]
    assert checkpoint["media_chunks"] == 1
    assert checkpoint["media_tokens"] >= 256
    assert warm.body["timings"]["prompt_n"] < 32

    different = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": media_prefix + suffix,
            "multimodal_data": [image_truck],
        },
        "id_slot": 2,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert different.status_code == 200
    assert different.body["timings"].get("checkpoint", {}).get("media_tokens", 0) == 0
    assert different.body["timings"]["prompt_n"] > warm.body["timings"]["prompt_n"] + 200

    ordered_prefix = "Compare A: <__media__> with B: <__media__>.\n" + "Check both regions. " * 20
    ordered = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": ordered_prefix,
            "multimodal_data": [image_cat, image_truck],
        },
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert ordered.status_code == 200

    reversed_media = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": ordered_prefix + suffix,
            "multimodal_data": [image_truck, image_cat],
        },
        "id_slot": 2,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert reversed_media.status_code == 200
    assert reversed_media.body["timings"].get("checkpoint", {}).get("media_tokens", 0) == 0

    ordered_warm = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": ordered_prefix + suffix,
            "multimodal_data": [image_cat, image_truck],
        },
        "id_slot": 3,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert ordered_warm.status_code == 200
    assert ordered_warm.body["timings"]["checkpoint"]["media_chunks"] == 2
    assert ordered_warm.body["timings"]["checkpoint"]["media_tokens"] >= 512

    encoding_prefix = "Inspect this encoded image carefully. " * 20
    encoding_cold = server.make_request("POST", "/chat/completions", data={
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_cat}},
            {"type": "text", "text": encoding_prefix},
        ]}],
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
        "max_tokens": 4,
    })
    assert encoding_cold.status_code == 200

    encoding_warm = server.make_request("POST", "/chat/completions", data={
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_cat}},
            {"type": "text", "text": encoding_prefix + suffix},
        ]}],
        "id_slot": 2,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
        "max_tokens": 4,
    })
    assert encoding_warm.status_code == 200
    assert encoding_warm.body["timings"]["checkpoint"]["media_tokens"] >= 256

    concurrent_prefix = "Inspect this concurrent image: <__media__>\n" + "State one detail. " * 20

    def concurrent_request(id_slot):
        return server.make_request("POST", "/completions", data={
            "prompt": {
                "prompt_string": concurrent_prefix,
                "multimodal_data": [image_cat],
            },
            "id_slot": id_slot,
            "cache_prompt": True,
            "temperature": 0,
            "top_k": 1,
            "return_tokens": True,
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(concurrent_request, [0, 1]))
    assert all(response.status_code == 200 for response in concurrent)
    assert concurrent[0].body["tokens"] == concurrent[1].body["tokens"]
    assert any(
        response.body["timings"].get("checkpoint", {}).get("media_tokens", 0) >= 256
        for response in concurrent
    )

    timed_out = False
    try:
        server.make_request("POST", "/completions", data={
            "prompt": {
                "prompt_string": media_prefix + " Continue." * 1000,
                "multimodal_data": [image_cat],
            },
            "id_slot": 1,
            "cache_prompt": True,
            "temperature": 0,
        }, timeout=0.01)
    except requests.exceptions.ReadTimeout:
        timed_out = True
    assert timed_out
    time.sleep(1)

    repeated = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": media_prefix + suffix,
            "multimodal_data": [image_cat],
        },
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert repeated.status_code == 200

    republished = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": media_prefix,
            "multimodal_data": [image_cat],
        },
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert republished.status_code == 200

    reattached = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": media_prefix + suffix,
            "multimodal_data": [image_cat],
        },
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert reattached.status_code == 200
    assert reattached.body["timings"]["checkpoint"]["media_tokens"] >= 256

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200

    def metric_value(name):
        match = re.search(rf"^llamacpp:{re.escape(name)} ([0-9.eE+-]+)$", metrics.body, re.MULTILINE)
        assert match is not None
        return float(match.group(1))

    assert metric_value("global_checkpoint_hits_total") > 0
    assert metric_value("global_checkpoint_coalesced_total") > 0
    assert metric_value("global_checkpoint_media_chunks_avoided_total") > 0
    assert metric_value("global_checkpoint_media_tokens_avoided_total") >= 256
    assert metric_value("global_checkpoint_media_bytes") > 0
    assert metric_value("global_checkpoint_evictions_total") > 0


def test_global_checkpoint_audio_hit_and_miss(audio_mmproj_server):
    server = audio_mmproj_server
    server.n_slots = 4
    server.n_ctx = 8192
    server.n_batch = 512
    server.n_ubatch = 512
    server.n_predict = 2
    server.kv_unified = True
    server.server_continuous_batching = True
    server.server_metrics = True
    server.ctx_checkpoints = 8
    server.checkpoint_min_step = 8
    audio_a = _get_wav_base64(440.0)
    audio_b = _get_wav_base64(880.0)
    prefix = "Transcribe this sound: <__media__>\n" + "Listen carefully. " * 20
    suffix = "Now give only the result."

    server.start(timeout_seconds=180)
    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    assert props.body["modalities"]["audio"] is True

    cold = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": prefix,
            "multimodal_data": [audio_a],
        },
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert cold.status_code == 200

    warm = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": prefix + suffix,
            "multimodal_data": [audio_a],
        },
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert warm.status_code == 200
    checkpoint = warm.body["timings"]["checkpoint"]
    assert checkpoint["media_chunks"] == 1
    assert checkpoint["media_tokens"] > 0

    different = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": prefix + suffix,
            "multimodal_data": [audio_b],
        },
        "id_slot": 2,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert different.status_code == 200
    assert different.body["timings"].get("checkpoint", {}).get("media_tokens", 0) == 0
    assert different.body["timings"]["prompt_n"] > warm.body["timings"]["prompt_n"]


def test_global_checkpoint_invalidated_when_projector_reloads(mmproj_server):
    server = mmproj_server
    server.n_slots = 4
    server.n_ctx = 4096
    server.n_batch = 512
    server.n_ubatch = 512
    server.n_predict = 2
    server.kv_unified = True
    server.server_continuous_batching = True
    server.server_metrics = True
    server.ctx_checkpoints = 8
    server.checkpoint_min_step = 8
    server.sleep_idle_seconds = 1
    image_cat = _get_img_base64(IMG_URL_CAT)
    prefix = "Describe this image: <__media__>\n" + "Check every detail. " * 20

    server.start()
    cold = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": prefix,
            "multimodal_data": [image_cat],
        },
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert cold.status_code == 200

    deadline = time.time() + 10
    while time.time() < deadline:
        props = server.make_request("GET", "/props")
        assert props.status_code == 200
        if props.body["is_sleeping"]:
            break
        time.sleep(0.1)
    else:
        pytest.fail("server did not unload the projector")

    reloaded = server.make_request("POST", "/completions", data={
        "prompt": {
            "prompt_string": prefix + "Answer briefly.",
            "multimodal_data": [image_cat],
        },
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
    })
    assert reloaded.status_code == 200
    assert reloaded.body["timings"].get("checkpoint", {}).get("media_tokens", 0) == 0
    assert reloaded.body["timings"]["prompt_n"] > 256


def test_slot_save_restore_with_image(mmproj_server):
    server = mmproj_server
    # Use the full SWA cache so the restored image prefix can be reused.
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content_cat = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert res.body["timings"]["cache_n"] == 0
    assert prompt_n_full > 32  # text plus image tokens are all processed

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    n_written = res.body["n_written"]
    assert n_saved > 0
    assert n_written > 0

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert res.body["n_read"] == n_written

    # a different image must not reuse the restored image tokens; only the text prefix before the image is common
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_TRUCK)],
        },
    })
    assert res.status_code == 200
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n < 16
    assert res.body["timings"]["prompt_n"] == prompt_n_full - cache_n

    # restore again and resend the same image: the image tokens must be reused and greedy sampling must reproduce the original content
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content_cat


def test_slot_save_restore_with_two_images(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.n_ctx = 2048  # two images need more than the default 512 per slot
    server.start()

    prompt = {
        "prompt_string": "A: <__media__> B: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT), _get_img_base64(IMG_URL_TRUCK)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 64

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    assert res.body["content"] == content


def test_slot_save_restore_with_image_across_restart(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # restart the server with the same model and mmproj: the saved file must restore in the new process and the image KV must be reused
    server.stop()
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content


def test_slot_save_restore_image_payload_larger_than_context(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    # the slot context, as the server computed it (n_ctx split across the slots)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    n_ctx_slot = res.body["default_generation_settings"]["n_ctx"]

    # a filler token, used to grow the prompt up to the slot context
    res = server.make_request("POST", "/tokenize", data={"content": " hello" * 8})
    assert res.status_code == 200
    assert len(res.body["tokens"]) == 8

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n" + " hello" * (n_ctx_slot - res.body["timings"]["prompt_n"] - 8),
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    path = os.path.join(server.slot_save_path, "mm_slot_large_payload.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    assert payload_size > n_ctx_slot  # the scenario under test: the payload does not fit in n_ctx

    # drop the image from the slot, then restore it from the file
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_media_file_without_mmproj(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 200

    # restart the same model without the mmproj: restoring the media file must fail gracefully and leave the slot usable
    server.stop()
    server.no_mmproj = True
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 400
    assert "Cannot restore media tokens without an mmproj" in res.body["error"]["message"]

    # A failed restore must leave the slot empty and usable.
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    content = res.body["content"]

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["content"] == content
