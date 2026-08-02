#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

function fail(msg) {
  console.error('ERROR: ' + msg);
  process.exit(1);
}

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  fail('Usage: node ua-tour-analyze.js <input.json> <output.json>');
}

let raw;
try {
  raw = fs.readFileSync(inputPath, 'utf8');
} catch (e) {
  fail('Could not read input file: ' + e.message);
}

let data;
try {
  data = JSON.parse(raw);
} catch (e) {
  fail('Could not parse input JSON: ' + e.message);
}

const nodes = Array.isArray(data.nodes) ? data.nodes : [];
const edges = Array.isArray(data.edges) ? data.edges : [];
const layers = Array.isArray(data.layers) ? data.layers : [];

const nodeById = new Map();
for (const n of nodes) {
  if (n && n.id) nodeById.set(n.id, n);
}

// Filter edges to those whose endpoints exist among the known nodes
// (allow function/class-level endpoints not present as top-level nodes;
// treat them loosely by mapping them to their containing file node when possible)
function resolveNodeRef(id) {
  if (nodeById.has(id)) return id;
  // try to map function:/class: refs like "function:scanner/cli.py:main" -> "file:scanner/cli.py"
  const m = /^(function|class):([^:]+):/.exec(id);
  if (m) {
    const fileId = 'file:' + m[2];
    if (nodeById.has(fileId)) return fileId;
  }
  return null;
}

const fanIn = new Map();
const fanOut = new Map();
const adjForward = new Map(); // for BFS: only imports/calls edges, forward direction, resolved to real node ids
const bidirectionalPairs = []; // [a,b] pairs with edges both ways among imports/calls

for (const n of nodes) {
  fanIn.set(n.id, 0);
  fanOut.set(n.id, 0);
  adjForward.set(n.id, new Set());
}

const edgeSet = new Set(); // "src|type|tgt" resolved, for bidirectional detection
const resolvedEdges = [];

for (const e of edges) {
  if (!e || !e.source || !e.target) continue;
  const src = resolveNodeRef(e.source);
  const tgt = resolveNodeRef(e.target);
  if (!src || !tgt) continue; // skip edges we can't map to known nodes
  resolvedEdges.push({ source: src, target: tgt, type: e.type });
  fanOut.set(src, (fanOut.get(src) || 0) + 1);
  fanIn.set(tgt, (fanIn.get(tgt) || 0) + 1);
  if (e.type === 'imports' || e.type === 'calls') {
    adjForward.get(src).add(tgt);
  }
  edgeSet.add(src + '|' + e.type + '|' + tgt);
}

// A. Fan-in ranking
const fanInRanking = nodes
  .map((n) => ({ id: n.id, fanIn: fanIn.get(n.id) || 0, name: n.name }))
  .sort((a, b) => b.fanIn - a.fanIn)
  .slice(0, 20);

// B. Fan-out ranking
const fanOutRanking = nodes
  .map((n) => ({ id: n.id, fanOut: fanOut.get(n.id) || 0, name: n.name }))
  .sort((a, b) => b.fanOut - a.fanOut)
  .slice(0, 20);

// C. Entry point candidates
const ENTRY_FILENAMES = new Set([
  'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js',
  'server.ts', 'server.js', 'mod.rs', 'main.go', 'main.py', 'main.rs',
  'manage.py', 'app.py', 'wsgi.py', 'asgi.py', 'run.py', '__main__.py',
  'Application.java', 'Main.java', 'Program.cs', 'config.ru', 'index.php',
  'App.swift', 'Application.kt', 'main.cpp', 'main.c',
]);

const fanOutValues = nodes.map((n) => fanOut.get(n.id) || 0).sort((a, b) => a - b);
const fanInValues = nodes.map((n) => fanIn.get(n.id) || 0).sort((a, b) => a - b);

function percentileThreshold(sortedArr, percentileFromTop) {
  if (sortedArr.length === 0) return 0;
  const idx = Math.max(0, Math.floor(sortedArr.length * (1 - percentileFromTop)));
  return sortedArr[idx];
}

const fanOutTop10Threshold = percentileThreshold(fanOutValues, 0.10);
const fanInBottom25Threshold = sortedForBottom(fanInValues, 0.25);

function sortedForBottom(sortedArr, percentile) {
  if (sortedArr.length === 0) return 0;
  const idx = Math.min(sortedArr.length - 1, Math.floor(sortedArr.length * percentile));
  return sortedArr[idx];
}

function isRootOrOneLevelDeep(filePath) {
  if (!filePath) return false;
  const parts = filePath.split('/').filter(Boolean);
  return parts.length <= 2;
}

const entryScored = [];
for (const n of nodes) {
  let score = 0;
  const fp = n.filePath || '';
  const baseName = n.name || path.basename(fp);

  if (n.type === 'document') {
    if (baseName === 'README.md' && isRootOrOneLevelDeep(fp)) {
      score += 5;
    } else if (/\.md$/i.test(baseName) && fp.split('/').filter(Boolean).length <= 1) {
      score += 2;
    }
  } else if (n.type === 'file') {
    if (ENTRY_FILENAMES.has(baseName)) score += 3;
    if (isRootOrOneLevelDeep(fp)) score += 1;
    const fo = fanOut.get(n.id) || 0;
    const fi = fanIn.get(n.id) || 0;
    if (fo >= fanOutTop10Threshold && fo > 0) score += 1;
    if (fi <= fanInBottom25Threshold) score += 1;
  }

  if (score > 0) {
    entryScored.push({ id: n.id, score, name: n.name, summary: n.summary });
  }
}

entryScored.sort((a, b) => b.score - a.score);
const entryPointCandidates = entryScored.slice(0, 5);

// D. BFS from top code entry point (skip documentation nodes)
const topCodeEntry = entryScored.find((e) => {
  const node = nodeById.get(e.id);
  return node && node.type !== 'document';
});

let bfsTraversal = { startNode: null, order: [], depthMap: {}, byDepth: {} };

if (topCodeEntry) {
  const start = topCodeEntry.id;
  const visited = new Set([start]);
  const order = [start];
  const depthMap = { [start]: 0 };
  const queue = [start];
  while (queue.length > 0) {
    const cur = queue.shift();
    const curDepth = depthMap[cur];
    const neighbors = Array.from(adjForward.get(cur) || []);
    for (const nb of neighbors) {
      if (!visited.has(nb)) {
        visited.add(nb);
        depthMap[nb] = curDepth + 1;
        order.push(nb);
        queue.push(nb);
      }
    }
  }
  const byDepth = {};
  for (const [id, d] of Object.entries(depthMap)) {
    if (!byDepth[d]) byDepth[d] = [];
    byDepth[d].push(id);
  }
  bfsTraversal = { startNode: start, order, depthMap, byDepth };
}

// E. Non-code file inventory
const nonCodeFiles = {
  documentation: [],
  infrastructure: [],
  data: [],
  config: [],
};

for (const n of nodes) {
  if (n.type === 'document') {
    nonCodeFiles.documentation.push({ id: n.id, name: n.name, type: n.type, summary: n.summary });
  } else if (n.type === 'service' || n.type === 'pipeline' || n.type === 'resource') {
    nonCodeFiles.infrastructure.push({ id: n.id, name: n.name, type: n.type, summary: n.summary });
  } else if (n.type === 'table' || n.type === 'schema' || n.type === 'endpoint') {
    nonCodeFiles.data.push({ id: n.id, name: n.name, type: n.type, summary: n.summary });
  } else if (n.type === 'config') {
    nonCodeFiles.config.push({ id: n.id, name: n.name, type: n.type, summary: n.summary });
  }
}

// F. Tightly coupled clusters
// Find bidirectional pairs among imports/calls edges
const pairSet = new Set();
const clusterUnion = new Map(); // simple union-find
function find(x) {
  if (!clusterUnion.has(x)) clusterUnion.set(x, x);
  let root = x;
  while (clusterUnion.get(root) !== root) root = clusterUnion.get(root);
  clusterUnion.set(x, root);
  return root;
}
function union(a, b) {
  const ra = find(a);
  const rb = find(b);
  if (ra !== rb) clusterUnion.set(ra, rb);
}

const edgeCountBetween = new Map(); // "a|b" (sorted) -> count

for (const e of resolvedEdges) {
  if (e.type !== 'imports' && e.type !== 'calls') continue;
  const a = e.source, b = e.target;
  const key = [a, b].sort().join('||');
  edgeCountBetween.set(key, (edgeCountBetween.get(key) || 0) + 1);
}

for (const e of resolvedEdges) {
  if (e.type !== 'imports' && e.type !== 'calls') continue;
  const reverseExists = resolvedEdges.some(
    (e2) => e2.source === e.target && e2.target === e.source && (e2.type === 'imports' || e2.type === 'calls')
  );
  if (reverseExists && e.source !== e.target) {
    union(e.source, e.target);
    pairSet.add([e.source, e.target].sort().join('||'));
  }
}

// Expand clusters: nodes connecting to 2+ existing cluster members
const rootGroups = new Map();
for (const n of nodes) {
  if (clusterUnion.has(n.id)) {
    const root = find(n.id);
    if (!rootGroups.has(root)) rootGroups.set(root, new Set());
    rootGroups.get(root).add(n.id);
  }
}

// Expansion pass
let changed = true;
let iterations = 0;
while (changed && iterations < 5) {
  changed = false;
  iterations++;
  for (const [root, members] of rootGroups.entries()) {
    for (const n of nodes) {
      if (members.has(n.id)) continue;
      let connections = 0;
      for (const e of resolvedEdges) {
        if (e.type !== 'imports' && e.type !== 'calls') continue;
        if ((e.source === n.id && members.has(e.target)) || (e.target === n.id && members.has(e.source))) {
          connections++;
        }
      }
      if (connections >= 2 && members.size < 5) {
        members.add(n.id);
        changed = true;
      }
    }
  }
}

const clusters = [];
for (const [root, members] of rootGroups.entries()) {
  const memberArr = Array.from(members);
  if (memberArr.length < 2) continue;
  let edgeCount = 0;
  for (const e of resolvedEdges) {
    if (memberArr.includes(e.source) && memberArr.includes(e.target) && e.source !== e.target) {
      edgeCount++;
    }
  }
  clusters.push({ nodes: memberArr, edgeCount });
}
clusters.sort((a, b) => b.edgeCount - a.edgeCount);
const topClusters = clusters.slice(0, 10);

// G. Layer list
const layerList = {
  count: layers.length,
  list: layers.map((l) => ({ id: l.id, name: l.name, description: l.description })),
};

// H. Node summary index
const nodeSummaryIndex = {};
for (const n of nodes) {
  nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary };
}

const result = {
  scriptCompleted: true,
  entryPointCandidates,
  fanInRanking,
  fanOutRanking,
  bfsTraversal,
  nonCodeFiles,
  clusters: topClusters,
  layers: layerList,
  nodeSummaryIndex,
  totalNodes: nodes.length,
  totalEdges: edges.length,
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf8');
} catch (e) {
  fail('Could not write output file: ' + e.message);
}

console.log('Analysis complete. Wrote results to ' + outputPath);
process.exit(0);
