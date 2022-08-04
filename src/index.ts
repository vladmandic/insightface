import * as tf from '@tensorflow/tfjs';
import { Human, Config } from '@vladmandic/human/dist/human.esm-nobundle';
import { log } from './log';

const modelUrls: Array<string> = [
  'models/insightface-efficientnet-b0.json',
  'models/insightface-ghostnet-strides1.json',
  'models/insightface-ghostnet-strides2.json',
  'models/insightface-mobilenet-emore.json',
  'models/insightface-mobilenet-swish.json',
];
const models: Array<tf.GraphModel> = [];

const humanConfig: Partial<Config> = { // user configuration for human, used to fine-tune behavior
  backend: 'webgl' as const,
  cacheSensitivity: 0,
  async: true,
  debug: false,
  modelBasePath: 'node_modules/@vladmandic/human/models',
  filter: { enabled: false, equalization: false, flip: false },
  face: { enabled: true, detector: { rotation: true }, mesh: { enabled: true }, attention: { enabled: false }, iris: { enabled: false }, description: { enabled: false }, emotion: { enabled: false } },
  body: { enabled: false },
  hand: { enabled: false },
  object: { enabled: false },
  gesture: { enabled: false },
};

async function main() {
  const human = new Human(humanConfig);
  log({ tf: tf.version_core, human: human.version });
  for (const url of modelUrls) {
    const model = await tf.loadGraphModel(url);
    models.push(model);
  }
  await human.load();
  await human.warmup();
}

window.onload = main;
