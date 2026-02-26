# create-voice-agent

A CLI that scaffolds a **production-ready, containerized infrastructure** for real-time Voice AI Agents using **LiveKit**, **Python**, and **Docker**.

Stop fighting WebRTC UDP port exhaustion, Docker permission errors, and token handshake wiring — this tool generates a perfect boilerplate in seconds.

## Quick Start

```bash
npx create-voice-agent my-voice-app
```

You'll be prompted to choose:

1. **LLM Provider** — Ollama (local GPU) or OpenAI (cloud)
2. **Next.js Frontend** — optional browser UI with LiveKit Room components

Then:

```bash
cd my-voice-app
cp .env.example .env    # fill in your API keys
docker compose up --build
```

## What Gets Generated

```
my-voice-app/
├── docker-compose.yml          # LiveKit + Redis + Agent (+ Frontend)
├── .env.example                # Pre-configured environment variables
├── README.md                   # Project-specific documentation
├── python-agent/
│   ├── Dockerfile              # Non-root user (EACCES fix)
│   ├── requirements.txt        # livekit-agents + plugins
│   └── agent.py                # VoicePipelineAgent (VAD→STT→LLM→TTS)
└── frontend/                   # (if selected)
    ├── Dockerfile
    ├── package.json            # includes livekit-server-sdk
    ├── app/api/token/route.ts  # Secure AccessToken generation
    └── app/page.tsx            # LiveKit Room + voice visualizer
```

## Key Architectural Decisions

### WebRTC UDP Port Limiting
LiveKit's UDP port range is strictly limited to `50000-50050` to prevent `docker-proxy` from spawning thousands of processes and hanging the host OS.

### Token Handshake
The Next.js frontend includes a `/api/token` endpoint using `livekit-server-sdk` to generate `AccessToken` JWTs. The browser fetches this token before connecting to the LiveKit room.

### Non-Root Docker User
The Python agent Dockerfile creates a dedicated `agentuser` to prevent `EACCES` permission errors common on Linux hosts.

### Ollama GPU Routing
When Ollama is selected, the agent container uses `host.docker.internal` to reach the host's GPU-accelerated Ollama instance.

## Development

```bash
# Clone and install
git clone https://github.com/geladons/create-voice-agent.git
cd create-voice-agent
npm install

# Run locally
node bin/create-voice-agent.js my-test-project
```

## License

MIT

