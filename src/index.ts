import * as tf from '@tensorflow/tfjs';
import { Human, Config, Result, FaceResult } from '@vladmandic/human/dist/human.esm-nobundle';
import { log } from './log';

const modelUrls: Array<string> = [
  'models/insightface-efficientnet-b0.json',
  'models/insightface-ghostnet-strides1.json',
  'models/insightface-ghostnet-strides2.json',
  'models/insightface-mobilenet-emore.json',
  'models/insightface-mobilenet-swish.json',
];
const models: Array<tf.GraphModel> = [];

/*
  for (const url of modelUrls) {
    const model = await tf.loadGraphModel(url);
    models.push(model);
  }
  await human.load();
  await human.warmup();
}

window.onload = main;
*/

const humanConfig: Partial<Config> = { // user configuration for human, used to fine-tune behavior
  backend: 'webgl' as const,
  cacheSensitivity: 0,
  async: true,
  debug: false,
  modelBasePath: 'node_modules/@vladmandic/human/models',
  filter: { enabled: false, equalization: false, flip: false },
  face: { enabled: true, detector: { return: true, rotation: true, maxDetected: 50, iouThreshold: 0.01, minConfidence: 0.2 }, mesh: { enabled: true }, attention: { enabled: false }, iris: { enabled: false }, description: { enabled: true }, emotion: { enabled: false } },
  body: { enabled: false },
  hand: { enabled: false },
  object: { enabled: false },
  gesture: { enabled: false },
  segmentation: { enabled: false },
};

const human = new Human(humanConfig); // new instance of human

interface Face extends FaceResult {
  embedding: number[];
  fileName: string;
  embeddings: Record<string, number[]>;
}

const all: Array<Face[]> = []; // array that will hold all detected faces
// const db: Array<{ source: string, embedding: number[] }> = []; // array that holds all known faces

function title(msg) {
  (document.getElementById('title') as HTMLDivElement).innerHTML = msg;
}

async function selectFaceCanvas(face: Face) {
  // loop through all canvases that contain faces
  const canvases = document.getElementsByClassName('face');
  title(`analyzing similarities of ${canvases.length} faces`);

  for (const canvas of Array.from(canvases) as HTMLCanvasElement[]) {
    // calculate similarity from selected face to current one in the loop
    const current = all[canvas['tag'].sample][canvas['tag'].face];
    const similarity = human.similarity(face.embedding, current.embedding);
    canvas['tag'].similarity = similarity;
    await human.tf.browser.toPixels(current.tensor, canvas);
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    ctx.font = 'small-caps 1rem "CenturyGothic"';
    ctx.fillStyle = 'rgba(0, 0, 0, 1)';
    ctx.fillText(`${(100 * similarity).toFixed(1)}%`, 3, 23);
    ctx.fillStyle = 'rgba(255, 255, 255, 1)';
    ctx.fillText(`${(100 * similarity).toFixed(1)}%`, 4, 24);
    ctx.fillText(`${current.age}y ${(100 * (current.genderScore || 0)).toFixed(1)}% ${current.gender}`, 4, canvas.height - 6);
  }
  // sort all faces by similarity
  const sorted = document.getElementById('faces') as HTMLDivElement;
  [...Array.from(sorted.children)]
    .sort((a, b) => parseFloat(b['tag'].similarity) - parseFloat(a['tag'].similarity))
    .forEach((canvas) => sorted.appendChild(canvas));
  title('faces sorted by similarity');
}

async function analyzeResults(index: number, res: Result, fileName: string) {
  // all[index] = res.face;
  all[index] = [];
  for (const i in res.face) {
    if (!res.face[i].tensor) continue; // did not get valid results
    if ((res.face[i].faceScore || 0) < (human.config?.face?.detector?.minConfidence || 0)) continue; // face analysis score too low
    all[index][i] = { ...res.face[i], embedding: res.face[i].embedding as number[], fileName, embeddings: {} };
    all[index][i].embeddings['human-faceres'] = res.face[i].embedding as number[];
    const canvas = document.createElement('canvas');
    canvas['tag'] = { sample: index, face: i, source: fileName };
    canvas.width = 200;
    canvas.height = 200;
    canvas.className = 'face';
    canvas.title = `
      source: ${fileName}
      score: ${Math.round(100 * res.face[i].boxScore)}% detection ${Math.round(100 * res.face[i].faceScore)}% analysis
      age: ${res.face[i].age} years
      gender: ${Math.round(100 * (res.face[i].genderScore as number))}% ${res.face[i].gender}
    `.replace(/  /g, ' ');
    await human.tf.browser.toPixels(res.face[i].tensor, canvas);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.font = 'small-caps 1rem "CenturyGothic"';
    ctx.fillStyle = 'rgba(255, 255, 255, 1)';
    ctx.fillText(`${res.face[i].age}y ${(100 * (res.face[i].genderScore || 0)).toFixed(1)}% ${res.face[i].gender}`, 4, canvas.height - 6);
    (document.getElementById('faces') as HTMLDivElement).appendChild(canvas);
    canvas.addEventListener('click', (evt) => {
      selectFaceCanvas(all[evt.target?.['tag'].sample][evt.target?.['tag'].face]);
      log(evt.target?.['tag'], all[evt.target?.['tag'].sample][evt.target?.['tag'].face]);
    });
  }
}

async function analyzeImage(index, image, length) {
  const numFaces = all.reduce((prev, curr) => prev + curr.length, 0);
  title(`analyzing input images |  ${Math.round(100 * index / length)}% [${index}/${length}] | found ${numFaces} faces`);
  return new Promise((resolve) => {
    const img = new Image(128, 128);
    img.onload = () => { // must wait until image is loaded
      (document.getElementById('images') as HTMLDivElement).appendChild(img); // and finally we can add it
      human.detect(img).then((res) => {
        analyzeResults(index, res, image); // then wait until image is analyzed
        resolve(true);
      });
    };
    img.onerror = () => {
      log('analyzeImage error:', index + 1, image);
      resolve(false);
    };
    img.title = image;
    img.src = encodeURI(image);
  });
}

async function main() {
  log({ tf: tf.version_core, human: human.version, backend: tf.getBackend() });
  title('loading models');
  await human.load();
  for (const url of modelUrls) models.push(await tf.loadGraphModel(url));
  // @ts-ignore private property
  log({ loaded: models.map((model) => model.modelUrl) });

  title('enumerating input images');
  const res = await fetch('../assets/samples');
  const dir: string[] = (res && res.ok) ? await res.json() : [];
  const images: Array<string> = dir.filter((img) => img.endsWith('.jpg'));

  const t0 = human.now();
  for (let i = 0; i < images.length; i++) await analyzeImage(i, images[i], images.length);
  const t1 = human.now();
  const faces = all.reduce((prev, curr) => prev + curr.length, 0);
  log({ images: all.length, faces, time: Math.round(t1 - t0) });
  title(`extracted ${faces} faces from ${all.length} images in ${Math.round(t1 - t0)} ms`);
}

window.onload = main;
