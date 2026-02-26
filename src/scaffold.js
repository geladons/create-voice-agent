import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ejs from 'ejs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEMPLATES_DIR = path.join(__dirname, '..', 'templates');

/**
 * Recursively render all .ejs templates from a source directory into a target directory.
 */
function renderDir(srcDir, destDir, data) {
  const entries = fs.readdirSync(srcDir, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(srcDir, entry.name);
    if (entry.isDirectory()) {
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
export async function scaffold({ projectName, llmProvider, includeFrontend }) {
  const projectDir = path.resolve(process.cwd(), projectName);

  if (fs.existsSync(projectDir)) {
    console.error(`\n❌ Directory "${projectName}" already exists. Aborting.`);
    process.exit(1);
  }

  fs.mkdirSync(projectDir, { recursive: true });

  const data = { projectName, llmProvider, includeFrontend };

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

