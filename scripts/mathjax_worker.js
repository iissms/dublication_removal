#!/usr/bin/env node
/**
 * MathJax worker that converts TeX expressions into MathML.
 *
 * Each line of stdin must be a JSON object: { "id": number, "text": string, "display": boolean? }.
 * The worker responds with { "id": number, "mathml": string } or an { "id", "error" } payload.
 */

const readline = require('readline');
const mjAPI = require('mathjax-node');

mjAPI.config({
  MathJax: {
    TeX: {
      extensions: ['AMSmath.js', 'AMSsymbols.js', 'noErrors.js', 'noUndefined.js', 'mhchem.js'],
    },
    tex2jax: {
      inlineMath: [
        ['$', '$'],
        ['\\\(', '\\\)'],
      ],
      displayMath: [
        ['$$', '$$'],
        ['\\[', '\\]'],
      ],
    },
  },
});

mjAPI.start();

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

function respond(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function convert(math, display) {
  return new Promise((resolve, reject) => {
    mjAPI.typeset(
      {
        math,
        format: 'TeX',
        mml: true,
        svg: false,
        html: false,
        speakText: false,
        display: Boolean(display),
      },
      (data) => {
        if (data.errors) {
          reject(new Error(Array.isArray(data.errors) ? data.errors.join('; ') : data.errors));
          return;
        }
        if (!data.mml) {
          reject(new Error('No MathML returned'));
          return;
        }
        resolve(data.mml);
      },
    );
  });
}

let queue = Promise.resolve();

rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) {
    return;
  }
  queue = queue.then(async () => {
    let payload;
    try {
      payload = JSON.parse(trimmed);
    } catch (error) {
      respond({error: 'invalid_json', detail: String(error)});
      return;
    }
    const identifier = payload.id ?? null;
    const text = typeof payload.text === 'string' ? payload.text : '';
    const display = Boolean(payload.display);
    if (!text.trim()) {
      respond({id: identifier, mathml: ''});
      return;
    }
    try {
      const mathml = await convert(text, display);
      respond({id: identifier, mathml});
    } catch (error) {
      respond({id: identifier, error: 'conversion_failed', detail: String(error)});
    }
  });
});

rl.once('close', () => {
  queue.finally(() => process.exit(0));
});
