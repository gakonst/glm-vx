"""Deterministic forward-call budgets, not wall-clock performance assertions."""
import queue
import threading
from contextlib import contextmanager

import numpy as np
import pytest

from glm_vx.scheduler import BusyError, Scheduler


class GatedModel:
    config = {'vocab_size': 16, 'max_position_embeddings': 256, 'eos_token_id': []}

    def __init__(self, fail_token=None):
        self.calls = queue.Queue()
        self.permits = threading.Semaphore(0)
        self.unblocked = threading.Event()
        self.fail_token = fail_token

    def new_cache(self):
        return []

    def forward(self, token, cache):
        owner = cache[0] if cache else token
        self.calls.put((owner, len(cache), token))
        if not self.unblocked.is_set():
            assert self.permits.acquire(timeout=5), 'test did not release model call'
        if token == self.fail_token:
            raise RuntimeError('injected model failure')
        cache.append(token)
        logits = np.zeros(16, dtype=np.float32)
        logits[15] = 10
        return logits

    def expect(self, owner, position, token=None):
        assert self.calls.get(timeout=5) == (owner, position, owner if token is None else token)

    def advance(self):
        self.permits.release()

    def release_all(self):
        self.unblocked.set()
        self.permits.release()


@contextmanager
def running(**kwargs):
    model = GatedModel(kwargs.pop('fail_token', None))
    kwargs.setdefault("decode_prefill_tokens", 1)
    scheduler = Scheduler(model, **kwargs)
    try:
        yield scheduler, model
    finally:
        model.release_all()
        scheduler.close()
        assert scheduler.status()['active_requests'] == 0
        assert scheduler.status()['reserved_tokens'] == 0


def done(req):
    while True:
        event = req.events.get(timeout=5)
        if event['type'] == 'done':
            return event


def test_all_decoders_run_before_one_total_prefill_token():
    with running(prefill_chunk=16) as (scheduler, model):
        scheduler.submit([1], 20)
        model.expect(1, 0)
        scheduler.submit([2], 20)
        scheduler.submit([3] * 40, 2)
        scheduler.submit([4] * 40, 2)
        model.advance()

        model.expect(1, 1, 15)
        model.advance()
        model.expect(2, 0)
        model.advance()
        # Both decoders get a turn; the two long prompts share a single
        # prompt-token budget and alternate, so neither class starves.
        for turn in range(6):
            model.expect(1, turn + 2, 15)
            model.advance()
            model.expect(2, turn + 1, 15)
            model.advance()
            model.expect(3 + turn % 2, turn // 2)
            model.advance()


def test_idle_prefills_use_bounded_chunks_in_round_robin_order():
    with running(prefill_chunk=3) as (scheduler, model):
        # Queue all requests before the worker can choose its first request.
        with scheduler.lock:
            for owner in (1, 2, 3):
                scheduler.submit([owner] * 12, 1)
        for start in (0, 3):
            for owner in (1, 2, 3):
                for position in range(start, start + 3):
                    model.expect(owner, position)
                    model.advance()


def test_new_short_prompt_interrupts_idle_chunk_at_next_token_boundary():
    with running(prefill_chunk=16) as (scheduler, model):
        scheduler.submit([1] * 40, 2)
        model.expect(1, 0)
        short = scheduler.submit([2], 3)
        model.advance()
        model.expect(2, 0)
        model.advance()
        assert short.events.get(timeout=5)['type'] == 'token'
        model.expect(2, 1, 15)
        model.advance()
        model.expect(1, 1)
        model.advance()
        model.expect(2, 2, 15)
        model.advance()
        assert done(short)['reason'] == 'length'


def test_cancelled_prefill_releases_admission_without_another_forward():
    with running(max_sequences=2, token_budget=32, prefill_chunk=16) as (scheduler, model):
        decoder = scheduler.submit([1], 4)
        model.expect(1, 0)
        cancelled = scheduler.submit([2] * 20, 2)
        with pytest.raises(BusyError):
            scheduler.submit([3], 1)
        cancelled.cancelled.set()
        model.advance()
        model.expect(1, 1, 15)
        model.advance()
        assert done(cancelled)['reason'] == 'cancelled'
        assert cancelled.cache is None
        assert scheduler.status()['reserved_tokens'] == len(decoder.tokens) + decoder.max_tokens
        replacement = scheduler.submit([3], 1)
        model.release_all()
        assert done(replacement)['reason'] == 'length'
        assert done(decoder)['reason'] == 'length'
        assert all(call[0] != 2 for call in list(model.calls.queue))


@pytest.mark.parametrize('failed_prompt', [[2] * 5, [3, 2]])
def test_prefill_errors_release_budget_and_other_requests_continue(failed_prompt):
    with running(fail_token=2, prefill_chunk=16) as (scheduler, model):
        decoder = scheduler.submit([1], 8)
        model.expect(1, 0)
        failed = scheduler.submit(failed_prompt, 2)
        survivor = scheduler.submit([4, 4], 3)
        model.release_all()
        failure = done(failed)
        assert failure['reason'] == 'error'
        assert failure['error'] == 'injected model failure'
        assert failed.cache is None
        assert done(survivor)['reason'] == 'length'
        assert done(decoder)['reason'] == 'length'
        assert scheduler.status()['reserved_tokens'] == 0


def test_shutdown_cancels_queued_work_after_inflight_forward():
    with running(prefill_chunk=16) as (scheduler, model):
        active = scheduler.submit([1] * 40, 4)
        model.expect(1, 0)
        queued = scheduler.submit([2] * 40, 4)
        # Zero timeout deterministically observes the blocked model owner.
        with pytest.raises(TimeoutError, match='resources must remain open'):
            scheduler.close(timeout=0)
        with pytest.raises(BusyError, match='closed'):
            scheduler.submit([3], 1)
        model.advance()
        assert done(active)['reason'] == 'cancelled'
        assert done(queued)['reason'] == 'cancelled'
        scheduler.close()
        assert model.calls.empty()


def test_configured_prefill_budget_is_shared_across_long_requests():
    with running(prefill_chunk=16, decode_prefill_tokens=3) as (scheduler, model):
        scheduler.submit([1], 20)
        model.expect(1, 0)
        scheduler.submit([2] * 40, 2)
        scheduler.submit([3] * 40, 2)
        model.advance()
        for turn in range(4):
            model.expect(1, turn+1, 15); model.advance()
            for offset in range(3):
                model.expect(2+turn%2, (turn//2)*3+offset); model.advance()


def test_slow_consumer_has_fixed_queue_and_does_not_block_other_requests():
    with running(prefill_chunk=1, max_pending_events=3) as (scheduler, model):
        abandoned = scheduler.submit([1], 30)
        model.expect(1, 0)
        survivor = scheduler.submit([2], 1)
        model.release_all()
        assert done(survivor)['reason'] == 'length'
        with scheduler.lock:
            assert scheduler.lock.wait_for(lambda: scheduler.count == 0, timeout=3)
        # Two tokens and one terminal event; limit independent of max_tokens.
        assert abandoned.events.maxsize == abandoned.events.qsize() == 3
        assert done(abandoned)['reason'] == 'backpressure'
        assert abandoned.generated == 2
        assert abandoned.cache is None
        assert scheduler.status()['reserved_tokens'] == 0


def test_cancellation_sweeps_queued_request_before_other_long_prefills():
    with running(prefill_chunk=1, max_sequences=4) as (scheduler, model):
        scheduler.submit([1] * 30, 1)
        model.expect(1, 0)
        scheduler.submit([2] * 30, 1)
        cancelled = scheduler.submit([3] * 30, 1)
        scheduler.cancel(cancelled)
        model.advance()
        # Reaping happens before another indivisible model forward starts.
        model.expect(2, 0)
        assert done(cancelled)['reason'] == 'cancelled'
        assert cancelled.generated == 0
        assert scheduler.status()['active_requests'] == 2
