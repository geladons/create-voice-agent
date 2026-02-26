import { Command } from 'commander';
import { select, input, confirm } from '@inquirer/prompts';
import { execSync } from 'node:child_process';
import { scaffold } from './scaffold.js';

// ─── Ollama API Fetching ──────────────────────────────────────────────
async function fetchOllamaModels(ollamaIp = '127.0.0.1') {
  try {
    const url = `http://${ollamaIp}:11434/api/tags`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.models || []).map((m) => m.name);
  } catch {
    return [];
  }
}

// ─── Model Selection Prompt ───────────────────────────────────────────
async function promptModelName(models, defaultModel) {
  if (models.length > 0) {
    const choices = models.map((m) => ({ name: m, value: m }));
    choices.push({ name: '✏️  Enter custom model name', value: '__custom__' });
    const picked = await select({
      message: 'Select an Ollama model:',
      choices,
    });
    if (picked === '__custom__') {
      return input({ message: 'Enter model name:', default: defaultModel });
    }
    return picked;
  }
  return input({ message: 'Enter Ollama model name:', default: defaultModel });
}

// ─── Docker Pre-Flight Check ──────────────────────────────────────────
function checkDocker() {
  try {
    execSync('docker --version', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

async function ensureDocker() {
  if (checkDocker()) return true;

  const shouldInstall = await confirm({
    message:
      '⚠️  Docker is not installed. Would you like to attempt to install it automatically?\n   (Uses official get.docker.com script, may require sudo password)',
    default: true,
  });

  if (!shouldInstall) return false;

  console.log('\n📦 Installing Docker via get.docker.com...\n');
  try {
    execSync('curl -fsSL https://get.docker.com | sh', { stdio: 'inherit' });
  } catch (err) {
    console.error('\n⚠️  Docker installation failed:', err.message);
    return false;
  }

  return checkDocker();
}

// ─── Docker Auto-Start ────────────────────────────────────────────────
function autoStartDocker(projectName, deploymentType, modelName) {
  const cwd = projectName;
  console.log('\n🐳 Starting Docker containers...\n');

  try {
    // Copy .env.example to .env
    execSync('cp .env.example .env', { cwd, stdio: 'inherit' });

    if (deploymentType === 'standalone') {
      // Start ollama container first
      console.log('⏳ Starting Ollama container...');
      execSync('docker compose up -d ollama', { cwd, stdio: 'inherit' });

      // Wait for Ollama to be ready
      console.log('⏳ Waiting for Ollama to be ready...');
      let ready = false;
      for (let i = 0; i < 30; i++) {
        try {
          execSync('docker compose exec ollama ollama list', {
            cwd,
            stdio: 'pipe',
          });
          ready = true;
          break;
        } catch {
          execSync('sleep 2', { stdio: 'pipe' });
        }
      }
      if (!ready) {
        console.log('⚠️  Ollama container took too long to start. Pull model manually.');
      } else {
        // Pull the model
        console.log(`⏳ Pulling model "${modelName}" (this may take a while)...`);
        execSync(`docker compose exec ollama ollama pull ${modelName}`, {
          cwd,
          stdio: 'inherit',
        });
      }
    }

    // Start all services
    console.log('\n🚀 Starting all services...');
    execSync('docker compose up --build -d', { cwd, stdio: 'inherit' });
    console.log('\n✅ All services are running!');
    console.log('   LiveKit:  ws://localhost:7880');
    console.log('   Frontend: http://localhost:3000 (if included)');
  } catch (err) {
    console.error('\n⚠️  Docker auto-start failed:', err.message);
    console.log('   You can start manually with: docker compose up --build');
  }
}

// ─── CLI Entry Point ──────────────────────────────────────────────────
const program = new Command();

program
  .name('create-voice-agent')
  .description(
    'Scaffold a production-ready, containerized Voice AI Agent using LiveKit, Python, and Docker',
  )
  .version('1.0.0')
  .argument('<project-name>', 'Name of the project directory to create')
  .action(async (projectName) => {
    console.log('\n🎙️  create-voice-agent v1.0.0\n');
    console.log(`Scaffolding project: ${projectName}\n`);

    // ─── Fetch Ollama models in background ────────────────────────
    let ollamaModels = await fetchOllamaModels();
    if (ollamaModels.length > 0) {
      console.log(`✅ Ollama detected — ${ollamaModels.length} model(s) available\n`);
    } else {
      console.log('ℹ️  Ollama not detected on localhost (will ask later)\n');
    }

    // ─── Step 1: Setup Mode ───────────────────────────────────────
    const setupMode = await select({
      message: 'Select setup mode:',
      choices: [
        {
          name: '⚡ Auto Mode (1-Click Magic)',
          value: 'auto',
          description: 'Ollama + defaults + auto-start Docker',
        },
        {
          name: '🔧 Advanced Mode',
          value: 'advanced',
          description: 'Full control over every option',
        },
        {
          name: '📦 Fast Mode (Scaffold Only)',
          value: 'fast',
          description: 'Generate files and exit — no auto-start',
        },
      ],
    });

    let llmProvider = 'ollama';
    let deploymentType = null;
    let modelName = 'qwen3-vl:2b';
    let language = 'en';
    let includeFrontend = true;
    let shouldAutoStart = false;
    let ollamaIp = 'localhost';

    // ─── AUTO MODE ────────────────────────────────────────────────
    if (setupMode === 'auto') {
      llmProvider = 'ollama';
      includeFrontend = true;
      shouldAutoStart = true;

      deploymentType = await select({
        message: 'How do you want to run Ollama?',
        choices: [
          {
            name: '🖥️  Use Existing Local Ollama (host)',
            value: 'existing',
            description: 'Ollama is already running on your machine',
          },
          {
            name: '🐳 Install Standalone in Docker (CPU/GPU)',
            value: 'standalone',
            description: 'Ollama runs as a Docker service',
          },
        ],
      });

      if (deploymentType === 'existing') {
        ollamaIp = await input({
          message: 'Ollama IP address:',
          default: 'localhost',
        });
        // Re-fetch models if IP changed
        const fetchIp = ollamaIp === 'localhost' ? '127.0.0.1' : ollamaIp;
        ollamaModels = await fetchOllamaModels(fetchIp);
        modelName = await promptModelName(ollamaModels, 'qwen3-vl:2b');
      } else {
        modelName = 'qwen3-vl:2b';
        console.log(`\n📦 Standalone mode: will use model "${modelName}" (fastest CPU model)`);
      }
    }

    // ─── ADVANCED MODE ────────────────────────────────────────────
    if (setupMode === 'advanced') {
      llmProvider = await select({
        message: 'Which LLM Provider?',
        choices: [
          { name: 'Ollama (Local)', value: 'ollama' },
          { name: 'OpenAI (Cloud)', value: 'openai' },
        ],
      });

      if (llmProvider === 'ollama') {
        deploymentType = await select({
          message: 'How do you want to run Ollama?',
          choices: [
            {
              name: '🖥️  Use Existing Local Ollama (host)',
              value: 'existing',
              description: 'Ollama is already running on your machine',
            },
            {
              name: '🐳 Install Standalone in Docker (CPU/GPU)',
              value: 'standalone',
              description: 'Ollama runs as a Docker service',
            },
          ],
        });

        ollamaIp = await input({
          message: 'Ollama IP address:',
          default: 'localhost',
        });

        if (deploymentType === 'existing') {
          const fetchIp = ollamaIp === 'localhost' ? '127.0.0.1' : ollamaIp;
          ollamaModels = await fetchOllamaModels(fetchIp);
          modelName = await promptModelName(ollamaModels, 'qwen3-vl:4b');
        } else {
          modelName = await input({
            message: 'Enter Ollama model name:',
            default: 'qwen3-vl:4b',
          });
        }

        language = await input({
          message: 'Agent language (e.g., en, ru, de):',
          default: 'en',
        });
      }

      includeFrontend = await confirm({
        message: 'Include a Next.js frontend?',
        default: true,
      });

      if (llmProvider === 'ollama') {
        shouldAutoStart = await confirm({
          message: 'Auto-start Docker containers now?',
          default: true,
        });
      }
    }

    // ─── FAST MODE ────────────────────────────────────────────────
    if (setupMode === 'fast') {
      llmProvider = await select({
        message: 'Which LLM Provider?',
        choices: [
          { name: 'Ollama (Local)', value: 'ollama' },
          { name: 'OpenAI (Cloud)', value: 'openai' },
        ],
      });

      if (llmProvider === 'ollama') {
        deploymentType = await select({
          message: 'How do you want to run Ollama?',
          choices: [
            {
              name: '🖥️  Use Existing Local Ollama (host)',
              value: 'existing',
            },
            {
              name: '🐳 Install Standalone in Docker (CPU/GPU)',
              value: 'standalone',
            },
          ],
        });
      }

      includeFrontend = await confirm({
        message: 'Include a Next.js frontend?',
        default: true,
      });

      // Fast mode uses defaults for model, language, no auto-start
      modelName = deploymentType === 'standalone' ? 'qwen3-vl:2b' : 'qwen3-vl:4b';
      shouldAutoStart = false;
    }

    // ─── Scaffold ─────────────────────────────────────────────────
    await scaffold({
      projectName,
      llmProvider,
      includeFrontend,
      deploymentType,
      modelName,
      language,
      ollamaIp,
    });

    console.log(`\n✅ Project "${projectName}" created successfully!\n`);

    if (shouldAutoStart) {
      const dockerReady = await ensureDocker();
      if (dockerReady) {
        autoStartDocker(projectName, deploymentType, modelName);
      } else {
        console.log('⚠️  Auto-start skipped. Please install Docker manually and run:');
        console.log(`  cd ${projectName} && docker compose up --build\n`);
      }
    } else {
      console.log('Next steps:');
      console.log(`  cd ${projectName}`);
      console.log('  cp .env.example .env');
      if (llmProvider === 'openai') {
        console.log('  # Fill in your API keys in .env');
      } else {
        console.log('  # No external API keys required — 100% local!');
        if (deploymentType === 'existing') {
          console.log('  # Make sure Ollama is running on your host machine');
        }
      }
      console.log('  docker compose up --build\n');
    }
  });

program.parse();