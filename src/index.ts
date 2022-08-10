import * as tf from '@tensorflow/tfjs';
import { Human, Config, Result, FaceResult } from '@vladmandic/human/dist/human.esm-nobundle';
import { log } from './log';

const models: Array<{ name: string, model: tf.GraphModel, perf: number, i: number }> = [];

const humanConfig: Partial<Config> = { // user configuration for human, used to fine-tune behavior
  backend: 'webgl' as const,
  cacheSensitivity: 0,
  async: true,
  debug: false,
  modelBasePath: 'node_modules/@vladmandic/human/models',
  filter: { enabled: false, equalization: false, flip: false },
  face: { enabled: true,
    detector: { return: true, rotation: true, maxDetected: 50, iouThreshold: 0.01, minConfidence: 0.2 },
    // @ts-ignore hidden param
    scale: 1.4,
    mesh: { enabled: true },
    attention: { enabled: false },
    iris: { enabled: false },
    description: { enabled: true },
    emotion: { enabled: false },
    // mobilefacenet: { enabled: true, modelPath: 'https://vladmandic.github.io/human-models/models/mobilefacenet.json' }, // uncomment to enable human mobilefacenet instead of default human faceres model
    // insightface: { enabled: true, modelPath: 'https://vladmandic.github.io/insightface/models/insightface-mobilenet-swish.json' }, // uncomment to enable human implementation of insightface instead of default human faceres model
  },
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

function title(msg) {
  (document.getElementById('title') as HTMLDivElement).innerHTML = msg;
}

function distance(descriptor1: number[], descriptor2: number[]) { // eucliedan distance
  let sum = 0;
  for (let i = 0; i < descriptor1.length; i++) {
    const diff = descriptor1[i] - descriptor2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

async function selectFaceCanvas(face: Face) {
  // loop through all canvases that contain faces
  const canvases = document.getElementsByClassName('face');
  title(`analyzing similarities of ${canvases.length} faces`);

  for (const canvas of Array.from(canvases) as HTMLCanvasElement[]) {
    // calculate similarity from selected face to current one in the loop
    const current = all[canvas['tag'].sample][canvas['tag'].face];
    // const similarity = human.similarity(face.embedding, current.embedding);
    const dist = distance(face.embedding, current.embedding);
    canvas['tag'].distance = dist;
    await human.tf.browser.toPixels(current.tensor, canvas);
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    ctx.font = 'small-caps 1rem "CenturyGothic"';
    ctx.fillStyle = 'rgba(0, 0, 0, 1)';
    ctx.fillText(`distance ${(dist).toFixed(2)}`, 3, 23);
    ctx.fillStyle = 'rgba(255, 255, 255, 1)';
    ctx.fillText(`distance ${(dist).toFixed(2)}`, 4, 24);
    ctx.fillText(`${current.age}y ${(100 * (current.genderScore || 0)).toFixed(1)}% ${current.gender}`, 4, canvas.height - 6);
  }
  // sort all faces by similarity
  const sorted = document.getElementById('faces') as HTMLDivElement;
  [...Array.from(sorted.children)]
    .sort((a, b) => parseFloat(a['tag'].distance) - parseFloat(b['tag'].distance))
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
    all[index][i].embeddings['human-default'] = res.face[i].embedding as number[];
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

async function loadImages(url: string) {
  title('enumerating input images');
  const res = await fetch(url);
  const dir: string[] = (res && res.ok) ? await res.json() : [];
  const images: Array<string> = dir.filter((img) => img.endsWith('.jpg'));

  const t0 = human.now();
  for (let i = 0; i < images.length; i++) await analyzeImage(i, images[i], images.length);
  const t1 = human.now();
  const faces = all.reduce((prev, curr) => prev + curr.length, 0);
  log({ images: all.length, faces, time: Math.round(t1 - t0) });
  title(`extracted | ${faces} faces from ${all.length} images in ${Math.round(t1 - t0)} ms`);
}

function switchEmbeddings(model: string) {
  for (const img of all) {
    for (const face of img) face.embedding = face.embeddings[model];
  }
  log({ embeddings: model });
}

async function analyzeEmbeddings() {
  title(`calculating embeddings | ${models.length} models`);
  for (let i = 0; i < all.length; i++) {
    for (const face of all[i]) {
      const resize = tf.image.resizeBilinear(face.tensor as unknown as tf.Tensor3D, [112, 112]);
      const batch = tf.expandDims(resize, 0);
      for (const model of models) {
        const t0 = performance.now();
        const res = model.model.execute(batch) as tf.Tensor;
        if (model.i > 0) model.perf += (performance.now() - t0);
        model.i++;
        face.embeddings[model.name] = Array.from(await res.data());
      }
      tf.dispose([resize, batch]);
    }
    title(`calculating embeddings | ${Math.round(100 * i / all.length)}% [${i}/${all.length}]`);
  }
  let total = { ms: 0, i: 0, html: '<option value="human-default" default>human-default</option>' };
  for (const model of models) {
    log({ model: model.name, total: model.perf, avg: model.perf / (model.i - 1), count: model.i });
    total = {
      ms: total.ms + model.perf,
      i: total.i + model.i,
      html: total.html + `<option value="${model.name}">${model.name}</option>`,
    };
  }
  const select = (document.getElementById('embeddings') as HTMLSelectElement);
  select.innerHTML = total.html;
  select.onchange = () => switchEmbeddings(select.value);
  title(`calculated embeddings | ${all.reduce((prev, curr) => prev + curr.length, 0)} faces in ${Math.round(total.ms)} ms`);
}

async function loadModels(url: string) {
  title('enumerating input images');
  const res = await fetch(url);
  const dir: string[] = (res && res.ok) ? await res.json() : [];
  const modelUrls: Array<string> = dir.filter((m) => m.endsWith('.json'));
  for (const m of modelUrls) {
    const name = m.replace(/^.*[\\/]/, '').replace('.json', '');
    const model = await tf.loadGraphModel(m);
    models.push({ name, model, perf: 0, i: 0 });
  }
  // @ts-ignore private property
  log({ models: models.map((model) => model.modelUrl) });
}

async function main() {
  log({ tf: tf.version_core, human: human.version, backend: tf.getBackend() });
  title('loading models');
  await human.load();
  await loadModels('../models');
  await loadImages('../assets/samples');
  await analyzeEmbeddings();
}

window.onload = main;
