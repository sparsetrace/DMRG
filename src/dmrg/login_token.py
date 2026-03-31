### login_token.py
from __future__ import annotations

import os
from typing import Iterable, Optional


def _load_local_dotenv() -> None:
    """
    Best-effort .env loading for local development.
    Safe even if python-dotenv is not installed.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv(override=False)


def _from_env(names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _from_colab(names: Iterable[str], *, set_env: bool = True) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None

    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            if set_env:
                os.environ[name] = value
            return value
    return None


def _from_kaggle(names: Iterable[str], *, set_env: bool = True) -> Optional[str]:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        return None

    client = UserSecretsClient()
    for name in names:
        try:
            value = client.get_secret(name)
        except Exception:
            value = None
        if value:
            if set_env:
                os.environ[name] = value
            return value
    return None


def login_token(
    name: str,
    *,
    aliases: Optional[Iterable[str]] = None,
    required: bool = True,
    set_env: bool = True,
    load_dotenv_first: bool = True,
    ) -> Optional[str]:
    """
    Load a secret/token from common environments.

    Search order:
      1) os.environ
      2) local .env (optional)
      3) Google Colab secrets
      4) Kaggle secrets

    Notes:
      - Modal / CI / Docker are usually already covered by os.environ.
      - If a token is found in Colab/Kaggle and set_env=True, it is copied
        into os.environ[name].

    Examples
    --------
    HF_TOKEN = login_token("HF_TOKEN", aliases=["HF_HUB_TOKEN"], required=False)
    WANDB_KEY = login_token("WANDB_API_KEY", required=False)
    """
    names = [name, *(aliases or [])]

    value = _from_env(names)
    if value:
        return value

    if load_dotenv_first:
        _load_local_dotenv()
        value = _from_env(names)
        if value:
            return value

    value = _from_colab(names, set_env=set_env)
    if value:
        if set_env:
            os.environ[name] = value
        return value

    value = _from_kaggle(names, set_env=set_env)
    if value:
        if set_env:
            os.environ[name] = value
        return value

    if required:
        tried = ", ".join(names)
        raise RuntimeError(
            f"Missing token. Tried [{tried}] in environment, .env, Colab secrets, and Kaggle secrets."
        )

    return None
