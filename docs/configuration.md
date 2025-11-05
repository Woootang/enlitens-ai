# Configuration Guide

## External Research Connectors

The multi-agent processor relies on the `ExternalResearchOrchestrator` to surface
fresh, verifiable citations from the public web. Configure the orchestrator by
setting the `ENLITENS_RESEARCH_CONNECTORS` environment variable to a JSON array
of connector definitions. If this variable is left empty the processor falls
back to the `NullConnector`, and startup validation will now abort to prevent
citation-free outputs.

Each connector entry must declare a `type` along with the fields required for
that connector family:

- `http` &mdash; call a REST or JSON API capable of answering knowledge queries.
  - Required: `endpoint`
  - Optional: `method` (`GET` by default), `headers`, `payload` (JSON template
    for POST requests), and `timeout`
- `mcp` &mdash; invoke an MCP client capable of running live research tools.
  - Required: `command`
  - Optional: `args` (list of command-line arguments) and `timeout`
- `static` &mdash; provide deterministic fixtures for offline development. (Only
  recommended for tests.)

### Example: HTTP and MCP connectors

```bash
export ENLITENS_RESEARCH_CONNECTORS='[
  {
    "type": "http",
    "endpoint": "https://api.example.com/search",
    "method": "GET",
    "headers": {
      "Authorization": "Bearer ${API_KEY}",
      "Accept": "application/json"
    },
    "timeout": 15
  },
  {
    "type": "mcp",
    "command": "/usr/local/bin/mcp-client",
    "args": ["--profile", "enlitens-web"],
    "timeout": 25
  }
]'
```

For POST-based APIs supply a `payload` template. The orchestrator will merge the
query topic, location, tags, and result limit into this payload at runtime.

```bash
export ENLITENS_RESEARCH_CONNECTORS='[
  {
    "type": "http",
    "endpoint": "https://civic-data.example.com/query",
    "method": "POST",
    "headers": {"Content-Type": "application/json"},
    "payload": {"include_snippets": true},
    "timeout": 20
  }
]'
```

### Deployment checklist

1. Ensure credentials referenced in connector headers or arguments are available
   in the execution environment (for example via other environment variables or
   secret stores).
2. Test the configuration locally with `python process_multi_agent_corpus.py`
   to confirm the startup probe reports at least one active connector.
3. Monitor telemetry for `startup_probe` entries and address any
   `missing_external_citations` alerts by adjusting connector coverage or agent
   prompts.
