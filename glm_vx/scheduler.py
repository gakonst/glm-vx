"""Bounded continuous scheduling. One model owner, request-local compressed KV caches.

The CPU prototype interleaves sequences; it does not claim fused GPU batching.
Reserve prompt+generation tokens up front so admitted requests cannot exhaust
this logical token budget halfway through decoding. Byte budgets remain a
backend deployment concern; token budget is NOT a GPU memory fit guarantee.
"""
from collections import deque
from dataclasses import dataclass, field
import queue
import threading
import time
import uuid
import numpy as np
from .prefix import PrefixCache

class BusyError(Exception): pass

@dataclass
class Request:
    tokens: tuple
    max_tokens: int
    temperature: float = 0.0
    seed: int = 0
    id: str = field(default_factory=lambda: "cmpl-" + uuid.uuid4().hex)
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    cache: object = None
    cursor: int = 0
    generated: int = 0
    next_token: int | None = None
    created: float = field(default_factory=time.monotonic)
    first_token_at: float | None = None
    rng: object = None
    cached_tokens: int = 0
    finished: bool = False
    reservation: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    prefix_seconds: float = 0.0

class Scheduler:
    def __init__(self, model, *, max_sequences=8, token_budget=8192, prefill_chunk=16, decode_prefill_tokens=4, prefix_cache_bytes=0, max_pending_events=64):
        if min(max_sequences,token_budget,prefill_chunk,decode_prefill_tokens) < 1: raise ValueError("positive limits required")
        if type(max_pending_events) is not int or max_pending_events < 2: raise ValueError("at least two pending events required")
        self.max_pending_events = max_pending_events
        self.prefix_cache = PrefixCache(model, prefix_cache_bytes)
        self.model = model
        self.max_sequences = max_sequences
        self.token_budget = token_budget
        self.prefill_chunk = prefill_chunk
        self.decode_prefill_tokens = decode_prefill_tokens
        self.lock = threading.Condition()
        self.pending = deque()
        self.reserved = 0
        self.count = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="glm-vx-model", daemon=True)
        self.thread.start()

    def submit(self, tokens, max_tokens, temperature=0.0, seed=0):
        cfg = self.model.config
        if not isinstance(tokens,list) or not tokens or any(type(t) is not int or not 0 <= t < cfg['vocab_size'] for t in tokens):
            raise ValueError("prompt must be a nonempty array of valid token IDs")
        if type(max_tokens) is not int or max_tokens < 1: raise ValueError("max_tokens must be a positive integer")
        if not isinstance(temperature,(int,float)) or isinstance(temperature,bool) or not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if type(seed) is not int or seed < 0: raise ValueError("seed must be a nonnegative integer")
        amount = len(tokens) + max_tokens
        if amount > cfg['max_position_embeddings']: raise ValueError("request exceeds context limit")
        req = Request(tuple(tokens),max_tokens,float(temperature),seed)
        req.reservation = amount
        req.rng = np.random.default_rng(seed)
        # Buffer cannot grow unbounded when a client stops reading.
        req.events = queue.Queue(maxsize=min(max_tokens+2, self.max_pending_events))
        with self.lock:
            if self.closed: raise BusyError("scheduler is closed")
            if self.count >= self.max_sequences or self.reserved+amount > self.token_budget:
                raise BusyError("admission capacity exhausted")
            self.reserved += amount
            self.count += 1
            self.pending.append(req)
            self.lock.notify()
        return req

    def _finish(self, req, reason, error=None):
        if req.finished: return
        req.finished = True
        req.cache = None
        with self.lock:
            self.reserved -= req.reservation
            self.count -= 1
            self.lock.notify_all()
        # Keep one terminal slot even for an abandoned consumer.
        req.events.put_nowait({"type":"done","reason":reason,"error":error,
            "prompt_tokens":len(req.tokens),"completion_tokens":req.generated,
            "cached_prompt_tokens":req.cached_tokens,
            "timings":{"prefill_seconds":req.prefill_seconds,"decode_seconds":req.decode_seconds,
                       "prefix_seconds":req.prefix_seconds}})

    def _sample(self, logits, req):
        logits = np.asarray(logits,dtype=np.float32)
        if logits.shape != (self.model.config['vocab_size'],) or not np.all(np.isfinite(logits)): raise ValueError("nonfinite or invalid logits")
        if req.temperature == 0: return int(np.argmax(logits))
        scaled = (logits.astype(np.float64)-float(np.max(logits)))/req.temperature
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(req.rng.choice(len(probs),p=probs))

    def _step(self, req, prefill_limit):
        if self.closed or req.cancelled.is_set(): self._finish(req,"cancelled"); return False
        if req.events.qsize() >= req.events.maxsize-1:
            self._finish(req,"backpressure","client output queue is full"); return False
        hit_logits = None
        if req.cache is None:
            start = time.perf_counter()
            hit = self.prefix_cache.lookup(req.tokens)
            req.prefix_seconds += time.perf_counter()-start
            if hit is None:
                req.cache = self.model.new_cache()
            else:
                req.cache = hit.cache
                req.cursor = req.cached_tokens = hit.tokens
                hit_logits = hit.logits
        if req.cursor < len(req.tokens):
            end = min(req.cursor+prefill_limit,len(req.tokens))
            while req.cursor < end:
                if self.closed or req.cancelled.is_set(): self._finish(req,"cancelled"); return False
                start = time.perf_counter()
                prefill_token = getattr(self.model, "prefill_token", None)
                if req.cursor < len(req.tokens)-1 and callable(prefill_token):
                    prefill_token(req.tokens[req.cursor],req.cache)
                else:
                    logits = self.model.forward(req.tokens[req.cursor],req.cache)
                req.prefill_seconds += time.perf_counter()-start
                req.cursor += 1
                if req.cursor < end:
                    # A forward call is indivisible. Reconsider admissions at
                    # each token boundary, even during an otherwise idle chunk.
                    with self.lock:
                        if self.pending or self.closed:
                            return True
            if req.cursor < len(req.tokens): return True
            if self.closed or req.cancelled.is_set(): self._finish(req,"cancelled"); return False
            start = time.perf_counter()
            self.prefix_cache.store(req.tokens, req.cache, logits)
            req.prefix_seconds += time.perf_counter()-start
        elif hit_logits is not None:
            logits = hit_logits
        else:
            start = time.perf_counter()
            logits = self.model.forward(req.next_token,req.cache)
            req.decode_seconds += time.perf_counter()-start
        if self.closed or req.cancelled.is_set(): self._finish(req,"cancelled"); return False
        token = self._sample(logits,req)
        req.generated += 1
        req.next_token = token
        if req.first_token_at is None: req.first_token_at = time.monotonic()
        req.events.put_nowait({"type":"token","token_id":token})
        eos = self.model.config.get('eos_token_id',[])
        if isinstance(eos,int): eos = [eos]
        reason = "stop" if token in eos else "length" if req.generated >= req.max_tokens else None
        if reason: self._finish(req,reason); return False
        return True

    def _run(self):
        decoding, prefilling = deque(), deque()
        while True:
            with self.lock:
                while not decoding and not prefilling and not self.pending and not self.closed:
                    self.lock.wait()
                prefilling.extend(self.pending)
                self.pending.clear()
                if self.closed:
                    for req in (*decoding, *prefilling): req.cancelled.set()
                if self.closed and not decoding and not prefilling: return

            # Reap every cancelled admission at the next model boundary, even
            # if its ordinary round-robin prefill turn is far away.
            for requests in (decoding, prefilling):
                for _ in range(len(requests)):
                    req = requests.popleft()
                    if req.cancelled.is_set(): self._finish(req, "cancelled")
                    else: requests.append(req)

            # One turn for every ready decoder, followed by bounded *total*
            # prefill work, rather than a chunk for each prefilling request.
            # Rotating the prefill queue guarantees progress without letting
            # multiple long prompts multiply the gap between decode turns.
            work = list(decoding)
            decoding.clear()
            prefill_limit = min(self.decode_prefill_tokens, self.prefill_chunk) if work else self.prefill_chunk
            if prefilling:
                work.append(prefilling.popleft())
            for req in work:
                try:
                    if self._step(req, prefill_limit):
                        if req.cursor < len(req.tokens):
                            with self.lock:
                                # An arrival that interrupted this chunk gets
                                # its turn before the interrupted request.
                                prefilling.extend(self.pending)
                                self.pending.clear()
                            prefilling.append(req)
                        else:
                            decoding.append(req)
                except Exception as exc:
                    self._finish(req,"error",str(exc))

    def status(self):
        with self.lock:
            return {"active_requests":self.count,"reserved_tokens":self.reserved,
                    "max_sequences":self.max_sequences,"token_budget":self.token_budget,
                    "max_pending_events":self.max_pending_events,"prefix_cache":self.prefix_cache.status()}

    def cancel(self, req):
        req.cancelled.set()
        with self.lock: self.lock.notify_all()

    def close(self, timeout=10):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError("model worker is still stopping; checkpoint resources must remain open")
        self.prefix_cache.clear()
