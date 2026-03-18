from threading import RLock
from typing import Any, Callable


_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = RLock()


def _normalize_key(cache_key: Any) -> str:
    if isinstance(cache_key, str):
        return cache_key
    return repr(cache_key)


def get_model(cache_key: Any) -> Any | None:
    key = _normalize_key(cache_key)
    with _CACHE_LOCK:
        return _MODEL_CACHE.get(key)


def set_model(cache_key: Any, model: Any) -> Any:
    key = _normalize_key(cache_key)
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = model
    return model


def get_or_create_model(cache_key: Any, factory: Callable[[], Any]) -> Any:
    key = _normalize_key(cache_key)

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    model = factory()

    with _CACHE_LOCK:
        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            return existing

        _MODEL_CACHE[key] = model
        return model


def clear_model_cache():
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
