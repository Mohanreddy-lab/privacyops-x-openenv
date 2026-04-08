"""FastAPI app for PrivacyOps-X."""

from __future__ import annotations

import os
import tempfile
import json
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.routing import APIRoute
from fastapi.responses import HTMLResponse
from starlette.responses import Response
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
    import openenv.core.env_server.web_interface as openenv_web_interface
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required to run PrivacyOps-X. Install dependencies first."
    ) from exc

try:
    from ..models import PrivacyOpsAction, PrivacyOpsObservation, PrivacyOpsState
    from .env import PrivacyOpsXEnvironment
    from .fixtures import load_tasks
except ImportError:  # pragma: no cover
    from models import PrivacyOpsAction, PrivacyOpsObservation, PrivacyOpsState
    from server.env import PrivacyOpsXEnvironment
    from server.fixtures import load_tasks


class TypedSchemaResponse(BaseModel):
    action: dict[str, Any]
    observation: dict[str, Any]
    state: dict[str, Any]


class DemoResponse(BaseModel):
    task: str
    steps: list[str]
    score: float


class EnvInfoResponse(BaseModel):
    env_name: str
    version: str
    tasks: list[str]
    max_steps: int
    reward_range: list[float]
    deterministic: bool


class HealthDetailResponse(BaseModel):
    status: str
    env_loaded: bool
    tasks_loaded: int


def _strip_frontmatter(markdown: str) -> str:
    lines = markdown.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return markdown
    try:
        closing_index = next(
            index for index in range(1, len(lines)) if lines[index].strip() == "---"
        )
    except StopIteration:
        return markdown
    return "\n".join(lines[closing_index + 1 :]).lstrip()


def _prepare_web_readme() -> None:
    root = Path(__file__).resolve().parent.parent
    source = root / "README.md"
    if not source.exists():
        return
    cleaned = _strip_frontmatter(source.read_text(encoding="utf-8"))
    target = Path(tempfile.gettempdir()) / "privacyops_x_web_readme.md"
    target.write_text(cleaned, encoding="utf-8")
    os.environ["ENV_README_PATH"] = str(target)

    original_loader = openenv_web_interface._load_readme_from_filesystem

    def _prefer_cleaned_readme(env_name: str | None) -> str | None:
        custom_path = os.environ.get("ENV_README_PATH")
        if custom_path and Path(custom_path).exists():
            try:
                return Path(custom_path).read_text(encoding="utf-8")
            except Exception:
                pass
        return original_loader(env_name)

    openenv_web_interface._load_readme_from_filesystem = _prefer_cleaned_readme


_prepare_web_readme()


app = create_app(
    PrivacyOpsXEnvironment,
    PrivacyOpsAction,
    PrivacyOpsObservation,
    env_name="privacyops_x",
    max_concurrent_envs=1,
)

app.router.routes = [
    route
    for route in app.router.routes
    if not (
        isinstance(route, APIRoute)
        and route.path in {"/", "/state", "/schema"}
        and "GET" in (route.methods or set())
    )
]


@app.middleware("http")
async def pretty_json_middleware(request: Request, call_next):
    response = await call_next(request)
    pretty = request.query_params.get("pretty")
    if not pretty or pretty.lower() in {"0", "false", "no"}:
        return response
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return response
    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    try:
        payload = json.loads(body)
    except Exception:
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=content_type,
            background=response.background,
        )
    pretty_bytes = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
    headers = dict(response.headers)
    headers.pop("content-length", None)
    return Response(
        content=pretty_bytes,
        status_code=response.status_code,
        headers=headers,
        media_type="application/json",
        background=response.background,
    )


@app.get(
    "/state",
    response_model=PrivacyOpsState,
    tags=["State Management"],
    summary="Get current environment state",
    description="Retrieve the typed PrivacyOps-X state model for the current environment instance.",
)
def state() -> PrivacyOpsState:
    env = PrivacyOpsXEnvironment()
    try:
        return env.state
    finally:
        env.close()


@app.get(
    "/schema",
    response_model=TypedSchemaResponse,
    tags=["Schema"],
    summary="Get typed JSON schemas",
    description="Return the typed PrivacyOps-X schemas for action, observation, and state.",
)
def schema() -> TypedSchemaResponse:
    return TypedSchemaResponse(
        action=PrivacyOpsAction.model_json_schema(),
        observation=PrivacyOpsObservation.model_json_schema(),
        state=PrivacyOpsState.model_json_schema(),
    )


@app.get(
    "/demo",
    response_model=DemoResponse,
    tags=["Environment Info"],
    summary="Get a demo trajectory summary",
    description="Return a short example path and score for quick evaluation.",
)
def demo() -> DemoResponse:
    return DemoResponse(
        task="easy_verified_access_with_injection",
        steps=["inspect_case", "open_record", "submit"],
        score=0.92,
    )


@app.get(
    "/envinfo",
    response_model=EnvInfoResponse,
    tags=["Environment Info"],
    summary="Get extended environment metadata",
    description="Return evaluation-oriented metadata for judges and tooling.",
)
def envinfo() -> EnvInfoResponse:
    tasks = load_tasks()
    step_limits = [int(task["step_limit"]) for task in tasks.values()]
    return EnvInfoResponse(
        env_name="PrivacyOps-X",
        version="1.0",
        tasks=["easy", "medium", "hard"],
        max_steps=max(step_limits) if step_limits else 0,
        reward_range=[0.0, 1.0],
        deterministic=True,
    )


@app.get(
    "/healthz",
    response_model=HealthDetailResponse,
    tags=["Health"],
    summary="Detailed health check",
    description="Extended health check with environment and task load info.",
)
def healthz() -> HealthDetailResponse:
    tasks = load_tasks()
    return HealthDetailResponse(
        status="healthy",
        env_loaded=True,
        tasks_loaded=len(tasks),
    )


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>PrivacyOps-X</title>
        <style>
          :root {
            --bg: #07131f;
            --panel: rgba(7, 18, 29, 0.88);
            --panel-soft: rgba(17, 35, 52, 0.84);
            --ink: #eef4f8;
            --muted: #a6b8c7;
            --line: rgba(132, 171, 196, 0.22);
            --teal: #72e6d1;
            --amber: #ffbf70;
            --rose: #ff8a80;
            --blue: #7db8ff;
            --shadow: 0 30px 90px rgba(0, 0, 0, 0.34);
          }
          body {
            margin: 0;
            min-height: 100vh;
            color: var(--ink);
            background:
              radial-gradient(circle at top left, rgba(114, 230, 209, 0.18), transparent 32%),
              radial-gradient(circle at 88% 14%, rgba(255, 191, 112, 0.16), transparent 24%),
              linear-gradient(160deg, #051019 0%, #081827 44%, #102338 100%);
            font-family: "Trebuchet MS", "Lucida Sans Unicode", sans-serif;
          }
          * {
            box-sizing: border-box;
          }
          .shell {
            width: min(1180px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 28px 0 48px;
          }
          .masthead {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 26px;
          }
          .brand {
            display: inline-flex;
            align-items: center;
            gap: 12px;
          }
          .brand-mark {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background:
              linear-gradient(135deg, rgba(114, 230, 209, 0.28), rgba(125, 184, 255, 0.18)),
              rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(125, 184, 255, 0.18);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
            font-size: 20px;
          }
          .brand h1 {
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(2rem, 4vw, 3rem);
            letter-spacing: 0.02em;
          }
          .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(114, 230, 209, 0.08);
            border: 1px solid rgba(114, 230, 209, 0.18);
            color: var(--teal);
            font-size: 0.83rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
          }
          .hero {
            position: relative;
            overflow: hidden;
            padding: 34px;
            border: 1px solid var(--line);
            border-radius: 30px;
            background:
              linear-gradient(145deg, rgba(13, 29, 43, 0.96), rgba(9, 21, 33, 0.9)),
              var(--panel);
            box-shadow: var(--shadow);
          }
          .hero::after {
            content: "";
            position: absolute;
            inset: auto -90px -120px auto;
            width: 280px;
            height: 280px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(125, 184, 255, 0.2), transparent 68%);
            pointer-events: none;
          }
          .hero-grid {
            display: grid;
            grid-template-columns: 1.4fr 0.9fr;
            gap: 28px;
            align-items: start;
          }
          .hero-copy p {
            margin: 14px 0 0;
            color: var(--muted);
            font-size: 1.06rem;
            line-height: 1.72;
            max-width: 64ch;
          }
          .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 24px;
          }
          .button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 13px 18px;
            border-radius: 14px;
            border: 1px solid transparent;
            text-decoration: none;
            font-weight: 700;
            transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
          }
          .button:hover {
            transform: translateY(-1px);
          }
          .button-primary {
            color: #04111d;
            background: linear-gradient(135deg, var(--teal), #95f4e4);
          }
          .button-secondary {
            color: var(--ink);
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.14);
          }
          .hero-panel {
            padding: 18px;
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: var(--panel-soft);
          }
          .hero-panel h2 {
            margin: 0 0 12px;
            font-size: 1rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--amber);
          }
          .hero-panel ul {
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
            line-height: 1.7;
          }
          .grid {
            display: grid;
            gap: 18px;
            margin-top: 24px;
          }
          .stats {
            grid-template-columns: repeat(4, minmax(0, 1fr));
          }
          .card {
            padding: 20px;
            border-radius: 22px;
            border: 1px solid var(--line);
            background: rgba(9, 21, 34, 0.82);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
          }
          .stat-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
          }
          .stat-value {
            margin-top: 8px;
            font-size: clamp(1.6rem, 3vw, 2.3rem);
            font-weight: 800;
          }
          .stat-value small {
            display: block;
            margin-top: 6px;
            font-size: 0.92rem;
            color: var(--muted);
            font-weight: 600;
          }
          .section-title {
            margin: 34px 0 14px;
            font-size: 0.92rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--amber);
          }
          .cases {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
          .case-card h3 {
            margin: 10px 0 10px;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.35rem;
          }
          .case-card p {
            margin: 0;
            color: var(--muted);
            line-height: 1.65;
          }
          .tag {
            display: inline-flex;
            align-items: center;
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(125, 184, 255, 0.1);
            color: var(--blue);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
          }
          .code-grid {
            grid-template-columns: 1.05fr 0.95fr;
          }
          .terminal {
            margin: 0;
            padding: 18px;
            overflow-x: auto;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: #04111d;
            color: #d5e6f8;
            font: 0.95rem/1.6 "Consolas", "Courier New", monospace;
          }
          .list {
            margin: 0;
            padding-left: 18px;
            color: var(--ink);
            line-height: 1.85;
            font-size: 1rem;
            font-weight: 600;
          }
          .list li {
            margin-bottom: 8px;
          }
          .list a {
            color: var(--teal);
            text-decoration: underline;
            text-underline-offset: 3px;
          }
          .list a:hover {
            color: #b7fff2;
          }
          .footer {
            margin-top: 28px;
            color: var(--muted);
            font-size: 0.92rem;
          }
          a {
            color: inherit;
          }
          code {
            padding: 2px 6px;
            border-radius: 7px;
            background: rgba(125, 184, 255, 0.08);
            color: #cfe6ff;
          }
          @media (max-width: 980px) {
            .hero-grid,
            .stats,
            .cases,
            .code-grid {
              grid-template-columns: 1fr;
            }
            .shell {
              width: min(100vw - 22px, 1180px);
            }
            .hero {
              padding: 24px;
              border-radius: 24px;
            }
          }
        </style>
      </head>
      <body>
        <div class="shell">
          <div class="masthead">
            <div class="brand">
              <div class="brand-mark">PX</div>
              <div>
                <div class="eyebrow">OpenEnv benchmark</div>
                <h1>PrivacyOps-X</h1>
              </div>
            </div>
          </div>

          <section class="hero">
            <div class="hero-grid">
              <div class="hero-copy">
                <div class="eyebrow">Safety-critical privacy operations</div>
                <p>
                  PrivacyOps-X evaluates whether an agent can handle real privacy
                  rights workflows under verification, retention, legal hold,
                  fraud, and audit constraints. It is built for benchmark-grade
                  scoring rather than toy interaction.
                </p>
                <div class="actions">
                  <a class="button button-primary" href="/web">Open Playground</a>
                  <a class="button button-secondary" href="/docs">API Docs</a>
                  <a class="button button-secondary" href="/schema">Typed Schema</a>
                  <a class="button button-secondary" href="/demo">Demo</a>
                  <a class="button button-secondary" href="/openapi.json">OpenAPI</a>
                </div>
              </div>
              <aside class="hero-panel">
                <h2>What judges can verify</h2>
                <ul>
                  <li>Typed <code>reset</code>, <code>step</code>, and <code>state</code> endpoints</li>
                  <li>Deterministic compliance, legal, and audit reviewers</li>
                  <li>Multi-turn requester interaction with revealed facts</li>
                  <li>Dense rewards plus final benchmark breakdowns</li>
                </ul>
              </aside>
            </div>
          </section>

          <section class="grid stats">
            <article class="card">
              <div class="stat-label">Scenarios</div>
              <div class="stat-value">3<small>easy, medium, hard</small></div>
            </article>
            <article class="card">
              <div class="stat-label">Reviewers</div>
              <div class="stat-value">3<small>compliance, legal, audit</small></div>
            </article>
            <article class="card">
              <div class="stat-label">Baseline</div>
              <div class="stat-value">1.0<small>easy / medium / hard on live validation</small></div>
            </article>
            <article class="card">
              <div class="stat-label">Deployment</div>
              <div class="stat-value">HF Space<small>dockerized and OpenEnv-valid</small></div>
            </article>
          </section>

          <div class="section-title">Judges quick links</div>
          <section class="grid">
            <article class="card">
              <ul class="list">
                <li><a href="/web">/web</a> interactive playground</li>
                <li><a href="/docs">/docs</a> Swagger UI</li>
                <li><a href="/schema">/schema</a> typed contracts</li>
                <li><a href="/demo">/demo</a> sample trajectory</li>
                <li><a href="/envinfo">/envinfo</a> evaluation metadata</li>
                <li><a href="/healthz">/healthz</a> detailed health</li>
              </ul>
            </article>
          </section>

          <div class="section-title">Benchmark cases</div>
          <section class="grid cases">
            <article class="card case-card">
              <span class="tag">Easy</span>
              <h3>Verified access with prompt injection</h3>
              <p>
                A California customer requests a data copy from the matched account
                email while trying to coerce the analyst into bypassing policy.
              </p>
            </article>
            <article class="card case-card">
              <span class="tag">Medium</span>
              <h3>Multi-account erasure with billing retention</h3>
              <p>
                A GDPR deletion request arrives from a mismatched sender and
                references two accounts, one of which carries statutory invoice
                retention obligations.
              </p>
            </article>
            <article class="card case-card">
              <span class="tag">Hard</span>
              <h3>Guardian request under legal hold and fraud review</h3>
              <p>
                A parent seeks access plus deletion for a minor account that is
                entangled with active fraud review and a legal hold.
              </p>
            </article>
          </section>

          <div class="section-title">Quick verification</div>
          <section class="grid code-grid">
            <article class="card">
              <pre class="terminal">curl -X POST /reset

curl -X POST /reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"medium_unverified_erasure_multi_account","seed":0}'

curl -X POST /step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"message_requester","content":"Please verify your identity and confirm which account emails are in scope."}}'</pre>
            </article>
            <article class="card">
              <ul class="list">
                <li><a href="/health">/health</a> confirms runtime readiness</li>
                <li><a href="/metadata">/metadata</a> exposes environment identity</li>
                <li><a href="/schema">/schema</a> publishes typed action, observation, and state contracts</li>
                <li><a href="/demo">/demo</a> shows a sample trajectory and score</li>
                <li><a href="/envinfo">/envinfo</a> provides judge-friendly metadata</li>
                <li><a href="/healthz">/healthz</a> returns detailed health info</li>
                <li><a href="/web">/web</a> opens the interactive OpenEnv playground</li>
                <li><a href="/docs">/docs</a> provides the FastAPI reference surface</li>
              </ul>
            </article>
          </section>

          <div class="footer">
            Designed for reproducible privacy-rights evaluation with deterministic
            reviewer engines, multi-turn requester interaction, and benchmark-grade
            final scoring.
          </div>
        </div>
      </body>
    </html>
    """


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
