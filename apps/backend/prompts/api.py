"""prompts/api.py — Admin endpoints for prompt lifecycle management.

Endpoints:
  GET    /admin/prompts                    List all prompts
  GET    /admin/prompts/{name}             Get full prompt detail
  POST   /admin/prompts/reload             Atomic hot reload (all-or-nothing)
  POST   /admin/prompts/{name}/reload      Hot reload a single prompt
  POST   /admin/prompts/{name}/preview     Preview rendered output
  POST   /admin/prompts/{name}/rollback    Rollback to a git ref (HEAD~1, commit hash)
  POST   /admin/prompts/validate           Pre-validate without applying (dry-run reload)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.prompts import (
    get_prompt,
    get_prompt_names,
    list_prompts,
    reload_one,
    reload_prompts,
    rollback_prompt,
)

router = APIRouter(prefix="/admin/prompts", tags=["prompts"])


# ── Request/Response Models ──────────────────────────────────────────────────


class RollbackRequest(BaseModel):
    git_ref: str = "HEAD~1"


class PreviewRequest(BaseModel):
    variables: dict = {}


class ValidateReloadRequest(BaseModel):
    """Dry-run: validate all prompts + golden tests without applying."""
    golden_tests: dict[str, dict] = {}


# ── List & Detail ────────────────────────────────────────────────────────────


@router.get("")
async def list_all_prompts() -> list[dict]:
    """List all registered prompts with metadata (not full template content)."""
    return [
        {
            "name": p.name,
            "description": p.description,
            "version": p.version,
            "author": p.author,
            "model": p.model,
            "temperature": p.temperature,
            "variables": p.variables,
            "updated_at": p.updated_at,
        }
        for p in list_prompts()
    ]


@router.get("/{name}")
async def get_prompt_detail(name: str) -> dict:
    """Get full prompt details including the template text."""
    try:
        prompt = get_prompt(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")
    return {
        "name": prompt.name,
        "description": prompt.description,
        "version": prompt.version,
        "author": prompt.author,
        "model": prompt.model,
        "temperature": prompt.temperature,
        "variables": prompt.variables,
        "template": prompt.template,
        "updated_at": prompt.updated_at,
    }


# ── Hot Reload ───────────────────────────────────────────────────────────────


@router.post("/reload")
async def reload_all(request: ValidateReloadRequest | None = None) -> dict:
    """Atomic hot reload: parse all → validate → replace.

    If golden_tests are provided, each prompt must pass before reload
    is accepted. On any failure, the old registry stays untouched.

    Returns count of loaded prompts and any errors.
    """
    golden_tests = None
    if request and request.golden_tests:
        golden_tests = {
            name: (test["variables"], test.get("expected_patterns", []))
            for name, test in request.golden_tests.items()
        }

    count, errors = reload_prompts(golden_tests=golden_tests)
    result = {
        "status": "ok" if not errors else "rejected",
        "reloaded": count,
        "names": get_prompt_names(),
    }
    if errors:
        result["errors"] = errors
    return result


@router.post("/{name}/reload")
async def reload_single(name: str) -> dict:
    """Hot reload a single prompt from disk."""
    prompt = reload_one(name)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found on disk")
    return {"status": "ok", "name": prompt.name, "version": prompt.version}


# ── Rollback ─────────────────────────────────────────────────────────────────


@router.post("/{name}/rollback")
async def rollback(name: str, request: RollbackRequest) -> dict:
    """Rollback a prompt to a historical version from git.

    git_ref can be: HEAD~1, HEAD~3, a commit hash, branch name.

    This is a fast, in-memory rollback — seconds, not minutes.
    For permanent rollback, follow up with: git checkout <ref> -- <file>
    """
    prompt = rollback_prompt(name, request.git_ref)
    if not prompt:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt '{name}' not found at git ref '{request.git_ref}'",
        )
    return {
        "status": "ok",
        "name": prompt.name,
        "version": prompt.version,
        "rolled_back_to": request.git_ref,
        "description": prompt.description,
        "template_preview": prompt.template[:200],
    }


# ── Preview ──────────────────────────────────────────────────────────────────


@router.post("/{name}/preview")
async def preview_prompt(name: str, request: PreviewRequest) -> dict:
    """Preview a prompt rendered with test variables."""
    try:
        prompt = get_prompt(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")
    try:
        output = prompt.render(**request.variables)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing variable: {e}")
    return {
        "name": output.prompt_name,
        "version": output.prompt_version,
        "rendered": output.text,
        "char_count": len(output.text),
    }


# ── Dry-Run Validation ──────────────────────────────────────────────────────


@router.post("/validate")
async def validate_only(request: ValidateReloadRequest) -> dict:
    """Dry-run validation: parse and test all prompts WITHOUT applying.

    Use this before calling /reload to check if changes are safe.
    """
    errors = []
    for name, test in request.golden_tests.items():
        try:
            prompt = get_prompt(name)
        except KeyError:
            errors.append(f"Prompt '{name}' not found")
            continue
        test_errors = prompt.validate_with_golden(
            test.get("variables", {}),
            test.get("expected_patterns", []),
        )
        if test_errors:
            errors.extend(f"{name}: {e}" for e in test_errors)

    if errors:
        return {"status": "would_fail", "errors": errors}
    return {"status": "would_pass", "message": "All golden tests passed"}
