import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ejs from 'ejs';
import os from 'node:os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEMPLATES_DIR = path.join(__dirname, '..', 'templates');

/**
 * Directories that should only be included for the local/ollama path.
 */
const LOCAL_ONLY_DIRS = ['custom_tts', 'custom_stt'];

/**
 * Recursively render all .ejs templates from a source directory into a target directory.
 */
function renderDir(srcDir, destDir, data) {
  const entries = fs.readdirSync(srcDir, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(srcDir, entry.name);
    if (entry.isDirectory()) {
      // Skip local-only directories when using cloud provider
      if (LOCAL_ONLY_DIRS.includes(entry.name) && data.llmProvider !== 'ollama') {
        continue;
      }
      const childDest = path.join(destDir, entry.name);
      fs.mkdirSync(childDest, { recursive: true });
      renderDir(srcPath, childDest, data);
    } else if (entry.name.endsWith('.ejs')) {
      const outName = entry.name.replace(/\.ejs$/, '');
      const template = fs.readFileSync(srcPath, 'utf-8');
      const rendered = ejs.render(template, data);
      fs.writeFileSync(path.join(destDir, outName), rendered);
    } else {
      // Copy non-template files as-is
      fs.copyFileSync(srcPath, path.join(destDir, entry.name));
    }
  }
}

/**
 * Main scaffolding function.
 */
export async function scaffold({
  projectName,
  llmProvider,
  includeFrontend,
  deploymentType = null,
  modelName = 'qwen3-vl:2b',
  language = 'en',
  ollamaIp = 'localhost',
  hostIp = '127.0.0.1',
}) {
  const projectDir = path.resolve(process.cwd(), projectName);

  if (fs.existsSync(projectDir)) {
    console.error(`\n❌ Directory "${projectName}" already exists. Aborting.`);
    process.exit(1);
  }

  fs.mkdirSync(projectDir, { recursive: true });

  const totalCpus = os.cpus().length;
  let cpuLimit;
  if (totalCpus <= 4) {
    cpuLimit = Math.max(1, totalCpus - 1);
  } else if (totalCpus <= 8) {
    cpuLimit = totalCpus - 2;
  } else {
    cpuLimit = 10;
  }
  // Ensure it's formatted as a string for Docker Compose (e.g., '6.0')
  const formattedCpuLimit = cpuLimit.toFixed(1);

  const data = {
    projectName,
    llmProvider,
    includeFrontend,
    deploymentType,
    modelName,
    language,
    ollamaIp,
    hostIp,
    cpuLimit: formattedCpuLimit,
  };

  // Render root-level templates
  const rootTemplates = ['docker-compose.yml.ejs', '.env.example.ejs', 'README.md.ejs'];
  for (const tpl of rootTemplates) {
    const template = fs.readFileSync(path.join(TEMPLATES_DIR, tpl), 'utf-8');
    const rendered = ejs.render(template, data);
    const outName = tpl.replace(/\.ejs$/, '');
    fs.writeFileSync(path.join(projectDir, outName), rendered);
  }

  // Render python-agent directory
  const agentSrc = path.join(TEMPLATES_DIR, 'python-agent');
  const agentDest = path.join(projectDir, 'python-agent');
  fs.mkdirSync(agentDest, { recursive: true });
  renderDir(agentSrc, agentDest, data);

  // Render frontend directory (if selected)
  if (includeFrontend) {
    const frontendSrc = path.join(TEMPLATES_DIR, 'frontend');
    const frontendDest = path.join(projectDir, 'frontend');
    fs.mkdirSync(frontendDest, { recursive: true });
    renderDir(frontendSrc, frontendDest, data);
  }

  console.log(`\n📁 Generated files in ./${projectName}/`);
}
