import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ejs from 'ejs';
import os from 'node:os';
import { execSync } from 'node:child_process';

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
  modelName = 'qwen2.5:1.5b',
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
  const calculatedLimit = totalCpus >= 12 ? 10 : totalCpus * 0.75;
  const cpuLimit = calculatedLimit.toFixed(1);

  const data = {
    projectName,
    llmProvider,
    includeFrontend,
    deploymentType,
    modelName,
    language,
    ollamaIp,
    hostIp,
    cpuLimit,
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

  // Pre-download models for offline operation
  if (llmProvider === 'ollama') {
    await preDownloadModels(projectDir, modelName, language, deploymentType);
  }
}

/**
 * Pre-download required models for offline operation
 */
async function preDownloadModels(projectDir, modelName, language, deploymentType) {
  console.log('\n📦 Pre-downloading models for offline operation...\n');

  try {
    // Create models directory
    const modelsDir = path.join(projectDir, 'python-agent', 'models');
    fs.mkdirSync(modelsDir, { recursive: true });

    // Download Piper TTS voice model
    const voiceModel = getVoiceModelForLanguage(language);
    console.log(`📥 Downloading Piper TTS voice model: ${voiceModel}`);
    await downloadPiperVoice(voiceModel, modelsDir);

    // Download Whisper STT model
    console.log('📥 Downloading Whisper STT model: base');
    await downloadWhisperModel('base', modelsDir);

    // For standalone deployment, also download the Ollama model
    if (deploymentType === 'standalone') {
      console.log(`📥 Downloading Ollama model: ${modelName}`);
      await downloadOllamaModel(modelName, projectDir);
    }

    console.log('\n✅ All models downloaded successfully for offline operation!');
  } catch (error) {
    console.warn('\n⚠️  Model pre-download failed, but scaffolding completed successfully.');
    console.warn('   Models will be downloaded on first run (requires internet connection).');
    console.warn(`   Error: ${error.message}`);
  }
}

/**
 * Get the appropriate Piper voice model for a language
 */
function getVoiceModelForLanguage(language) {
  const voiceModels = {
    'en': 'en_US-lessac-medium',
    'ru': 'ru_RU-dmitri-medium',
    'de': 'de_DE-thorsten-medium',
    'fr': 'fr_FR-upmc-medium',
    'es': 'es_ES-carlfm-medium'
  };
  return voiceModels[language] || 'en_US-lessac-medium';
}

/**
 * Download Piper TTS voice model
 */
async function downloadPiperVoice(ttsModel, modelsDir) {
  const { spawn } = await import('node:child_process');
  
  return new Promise((resolve, reject) => {
    const pythonScript = `
import urllib.request
import os
import sys

def build_piper_url(tts_model):
    parts = tts_model.split("-")
    if len(parts) != 3:
        raise ValueError(f"Unsupported Piper model format: {tts_model}")
    
    lang_dir = parts[0].split("_")[0]
    return f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{lang_dir}/{parts[0]}/{parts[1]}/{parts[2]}/{tts_model}"

def download_model(tts_model, models_dir):
    try:
        hf_url = build_piper_url(tts_model)
        onnx_path = os.path.join(models_dir, f"{tts_model}.onnx")
        json_path = os.path.join(models_dir, f"{tts_model}.onnx.json")
        
        print(f"Downloading {tts_model}...")
        urllib.request.urlretrieve(f"{hf_url}.onnx?download=true", onnx_path)
        urllib.request.urlretrieve(f"{hf_url}.onnx.json?download=true", json_path)
        print(f"Successfully downloaded {tts_model}")
    except Exception as e:
        print(f"Error downloading {tts_model}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    tts_model = sys.argv[1]
    models_dir = sys.argv[2]
    download_model(tts_model, models_dir)
`;

    const pythonProcess = spawn('python3', ['-c', pythonScript, ttsModel, modelsDir]);
    
    pythonProcess.stdout.on('data', (data) => {
      console.log(data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(data.toString());
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Python script exited with code ${code}`));
      }
    });
  });
}

/**
 * Download Whisper STT model
 */
async function downloadWhisperModel(modelName, modelsDir) {
  const { spawn } = await import('node:child_process');
  
  return new Promise((resolve, reject) => {
    const pythonScript = `
import os
import sys
from huggingface_hub import snapshot_download

def download_whisper_model(model_name, models_dir):
    try:
        cache_dir = os.path.join(models_dir, "faster-whisper")
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Downloading Whisper model: {model_name}")
        snapshot_download(
            repo_id=f"openai/whisper-{model_name}",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            cache_dir=cache_dir
        )
        print(f"Successfully downloaded Whisper {model_name}")
    except Exception as e:
        print(f"Error downloading Whisper {model_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    models_dir = sys.argv[2]
    download_whisper_model(model_name, models_dir)
`;

    const pythonProcess = spawn('python3', ['-c', pythonScript, modelName, modelsDir]);
    
    pythonProcess.stdout.on('data', (data) => {
      console.log(data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(data.toString());
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Python script exited with code ${code}`));
      }
    });
  });
}

/**
 * Download Ollama model for standalone deployment
 */
async function downloadOllamaModel(modelName, projectDir) {
  try {
    // For standalone deployment, we'll create a script that pulls the model
    const pullScript = `#!/bin/bash
echo "Pulling Ollama model: ${modelName}"
docker compose exec ollama ollama pull ${modelName}
echo "Model ${modelName} downloaded successfully"
`;

    const scriptPath = path.join(projectDir, 'download-models.sh');
    fs.writeFileSync(scriptPath, pullScript);
    fs.chmodSync(scriptPath, 0o755);
    
    console.log('📝 Created download-models.sh script for Ollama model');
  } catch (error) {
    throw new Error(`Failed to create Ollama download script: ${error.message}`);
  }
}
