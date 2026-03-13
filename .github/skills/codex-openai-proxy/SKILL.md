---
name: codex-openai-proxy
description: Configure proxy environment variables so Codex-based tooling can reach OpenAI endpoints. Use when users ask to enable HTTP/HTTPS proxy for Codex plugin, OpenAI API connectivity, or local proxy bootstrap.
---
# codex-openai-proxy

为 Codex 插件配置代理，使其访问 OpenAI 接口时继承明确的 HTTP/HTTPS proxy 环境变量。

## Execute

1. Confirm the target shell and platform because environment variable syntax differs across bash, PowerShell, and cmd.
2. Set both http_proxy and https_proxy to http://127.0.0.1:10792 for the current session.
3. Restart the Codex host process or VS Code if the plugin was already running before the proxy variables were set.
4. Verify that the effective shell environment exposes both variables before retrying the OpenAI request.
5. If connectivity still fails, verify that the local proxy process is listening on 127.0.0.1:10792 and that no conflicting NO_PROXY setting is bypassing the target domains.

## Commands

### bash / zsh

```bash
export http_proxy=http://127.0.0.1:10792
export https_proxy=http://127.0.0.1:10792
```

### PowerShell

```powershell
$env:http_proxy = "http://127.0.0.1:10792"
$env:https_proxy = "http://127.0.0.1:10792"
```

### cmd.exe

```cmd
set http_proxy=http://127.0.0.1:10792
set https_proxy=http://127.0.0.1:10792
```

## Verification

- bash / zsh: `echo $http_proxy` and `echo $https_proxy`
- PowerShell: `Get-ChildItem Env:http_proxy,Env:https_proxy`
- cmd.exe: `set http_proxy` and `set https_proxy`

## Repository Anchors

- `.github/skills/codex-openai-proxy/SKILL.md`

## Output Contract

- `shell_type`
- `proxy_http`
- `proxy_https`
- `restart_required`
- `verification_command`

## Notes

- The user-provided `export` syntax applies to bash-like shells.
- On Windows PowerShell, use `$env:http_proxy` and `$env:https_proxy` instead.
- Many extensions only inherit environment variables when the host process starts, so restarting the editor is often required.