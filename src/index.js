import { Command } from 'commander';
import { select } from '@inquirer/prompts';
import { scaffold } from './scaffold.js';

const program = new Command();

program
  .name('create-voice-agent')
  .description('Scaffold a production-ready, containerized Voice AI Agent using LiveKit, Python, and Docker')
  .version('1.0.0')
  .argument('<project-name>', 'Name of the project directory to create')
  .action(async (projectName) => {
    console.log(`\n🎙️  create-voice-agent v1.0.0\n`);
    console.log(`Scaffolding project: ${projectName}\n`);

    const llmProvider = await select({
      message: 'Which LLM Provider do you want to use?',
      choices: [
        { name: 'Ollama (Local)', value: 'ollama' },
        { name: 'OpenAI (Cloud)', value: 'openai' },
      ],
    });

    const includeFrontend = await select({
      message: 'Do you want to include a minimal Next.js frontend?',
      choices: [
        { name: 'Yes', value: true },
        { name: 'No', value: false },
      ],
    });

    await scaffold({
      projectName,
      llmProvider,
      includeFrontend,
    });

    console.log(`\n✅ Project "${projectName}" created successfully!\n`);
    console.log(`Next steps:`);
    console.log(`  cd ${projectName}`);
    console.log(`  cp .env.example .env`);
    if (llmProvider === 'openai') {
      console.log(`  # Fill in your API keys in .env`);
    } else {
      console.log(`  # Make sure Ollama is running on your host machine`);
      console.log(`  # No external API keys required — 100% local!`);
    }
    console.log(`  docker compose up --build\n`);
  });

program.parse();

