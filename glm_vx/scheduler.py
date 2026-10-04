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

class BusyError(Exception): pass

@dataclass
class Request:
    tokens: list
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

class Scheduler:
    def __init__(self, model, *, max_sequences=8, token_budget=8192, prefill_chunk=16):
        if min(max_sequences,token_budget,prefill_chunk) < 1: raise ValueError("positive limits required")
        self.model = model
        self.max_sequences = max_sequences
        self.token_budget = token_budget
        self.prefill_chunk = prefill_chunk
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
        req = Request(list(tokens),max_tokens,float(temperature),seed)
        req.rng = np.random.default_rng(seed)
        # Buffer cannot grow unbounded when a client stops reading.
        req.events = queue.Queue(maxsize=max_tokens+2)
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
        req.cache = None
        with self.lock:
            self.reserved -= len(req.tokens)+req.max_tokens
            self.count -= 1
            self.lock.notify_all()
        req.events.put_nowait({"type":"done","reason":reason,"error":error,
            "prompt_tokens":len(req.tokens),"completion_tokens":req.generated})

    def _sample(self, logits, req):
        logits = np.asarray(logits,dtype=np.float32)
        if logits.ndim != 1 or not np.all(np.isfinite(logits)): raise ValueError("nonfinite or invalid logits")
        if req.temperature == 0: return int(np.argmax(logits))
        scaled = (logits.astype(np.float64)-float(np.max(logits)))/req.temperature
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(req.rng.choice(len(probs),p=probs))

    def _step(self, req):
        if req.cancelled.is_set(): self._finish(req,"cancelled"); return False
        if req.cache is None: req.cache = self.model.new_cache()
        if req.cursor < len(req.tokens):
            end = min(req.cursor+self.prefill_chunk,len(req.tokens))
            while req.cursor < end:
                if req.cancelled.is_set(): self._finish(req,"cancelled"); return False
                logits = self.model.forward(req.tokens[req.cursor],req.cache)
                req.cursor += 1
            if req.cursor < len(req.tokens): return True
        else:
            logits = self.model.forward(req.next_token,req.cache)
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
        active = deque()
        while True:
            with self.lock:
                while not active and not self.pending and not self.closed: self.lock.wait()
                active.extend(self.pending)
                self.pending.clear()
                if self.closed:
                    for req in active: req.cancelled.set()
                if self.closed and not active: return
            req = active.popleft()
            try:
                if self._step(req): active.append(req)
            except Exception as exc:
                self._finish(req,"error",str(exc))

    def status(self):
        with self.lock:
            return {"active_requests":self.count,"reserved_tokens":self.reserved,
                    "max_sequences":self.max_sequences,"token_budget":self.token_budget}

    def close(self, timeout=10):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError("model worker is still stopping; checkpoint resources must remain open")
