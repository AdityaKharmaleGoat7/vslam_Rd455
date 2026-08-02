#!/usr/bin/env node
'use strict';

function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  if (!inPath || !outPath) {
    console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
    process.exit(1);
  }

  let data;
  try {
    data = JSON.parse(require('fs').readFileSync(inPath, 'utf8'));
  } catch (e) {
    console.error('Failed to read/parse input JSON: ' + e.message);
    process.exit(1);
  }

  const fs = require('fs');
  const path = require('path');

  const fileNodes = data.fileNodes || [];
  const importEdges = data.importEdges || [];
  const allEdges = data.allEdges || [];

  const nodeById = new Map();
  for (const n of fileNodes) nodeById.set(n.id, n);

  // A. Directory Grouping
  function commonPrefix(paths) {
    if (paths.length === 0) return '';
    const split = paths.map(p => p.split('/').slice(0, -1));
    let prefix = split[0];
    for (let i = 1; i < split.length; i++) {
      const s = split[i];
      let j = 0;
      while (j < prefix.length && j < s.length && prefix[j] === s[j]) j++;
      prefix = prefix.slice(0, j);
      if (prefix.length === 0) break;
    }
    return prefix.length ? prefix.join('/') + '/' : '';
  }

  const allPaths = fileNodes.map(n => n.filePath || n.name || '');
  const prefix = commonPrefix(allPaths);

  function groupFor(filePath) {
    let rest = filePath;
    if (prefix && rest.startsWith(prefix)) {
      rest = rest.slice(prefix.length);
    }
    const segs = rest.split('/').filter(Boolean);
    if (segs.length <= 1) {
      // flat: either at prefix root or single-segment file
      // try grouping by extension pattern
      const base = segs[0] || rest;
      if (/\.test\.|\.spec\.|^test_|_test\.go$|Test\.java$|_spec\.rb$|Test\.php$|Tests\.cs$/.test(base)) return 'test';
      if (/\.config\./.test(base)) return 'config';
      // fallback: use first dir segment of full path (before prefix) or root
      const fullSegs = filePath.split('/').filter(Boolean);
      if (fullSegs.length > 1) return fullSegs[0];
      return '(root)';
    }
    return segs[0];
  }

  const directoryGroups = {};
  for (const n of fileNodes) {
    const g = groupFor(n.filePath || n.name || '');
    if (!directoryGroups[g]) directoryGroups[g] = [];
    directoryGroups[g].push(n.id);
  }

  // B. Node Type Grouping
  const nodeTypeGroups = {};
  for (const n of fileNodes) {
    const t = n.type || 'file';
    if (!nodeTypeGroups[t]) nodeTypeGroups[t] = [];
    nodeTypeGroups[t].push(n.id);
  }

  // C. Import Adjacency + fan-in/out
  const fileFanOut = {};
  const fileFanIn = {};
  const adjacency = {};
  for (const n of fileNodes) {
    fileFanOut[n.id] = 0;
    fileFanIn[n.id] = 0;
    adjacency[n.id] = [];
  }
  for (const e of importEdges) {
    if (!nodeById.has(e.source) || !nodeById.has(e.target)) continue;
    fileFanOut[e.source] = (fileFanOut[e.source] || 0) + 1;
    fileFanIn[e.target] = (fileFanIn[e.target] || 0) + 1;
    adjacency[e.source].push(e.target);
  }

  const idToGroup = {};
  for (const [g, ids] of Object.entries(directoryGroups)) {
    for (const id of ids) idToGroup[id] = g;
  }

  // group-level import sets
  const groupImportsFrom = {}; // group -> set of groups it imports from
  const groupImportedBy = {}; // group -> set of groups that import it
  for (const g of Object.keys(directoryGroups)) {
    groupImportsFrom[g] = new Set();
    groupImportedBy[g] = new Set();
  }
  for (const e of importEdges) {
    const sg = idToGroup[e.source];
    const tg = idToGroup[e.target];
    if (!sg || !tg) continue;
    if (sg !== tg) {
      groupImportsFrom[sg].add(tg);
      groupImportedBy[tg].add(sg);
    }
  }

  // D. Cross-Category Dependency Analysis
  const crossCategoryMap = new Map();
  for (const e of allEdges) {
    const s = nodeById.get(e.source);
    const t = nodeById.get(e.target);
    if (!s || !t) continue; // only count edges where both endpoints are file-level nodes we know
    if (s.type === t.type) continue; // cross-category only (different node types)
    const key = s.type + '|' + t.type + '|' + e.type;
    crossCategoryMap.set(key, (crossCategoryMap.get(key) || 0) + 1);
  }
  const crossCategoryEdges = [];
  for (const [key, count] of crossCategoryMap.entries()) {
    const [fromType, toType, edgeType] = key.split('|');
    crossCategoryEdges.push({ fromType, toType, edgeType, count });
  }

  // E. Inter-Group Import Frequency
  const interGroupMap = new Map();
  for (const e of importEdges) {
    const sg = idToGroup[e.source];
    const tg = idToGroup[e.target];
    if (!sg || !tg || sg === tg) continue;
    const key = sg + '|' + tg;
    interGroupMap.set(key, (interGroupMap.get(key) || 0) + 1);
  }
  const interGroupImports = [];
  for (const [key, count] of interGroupMap.entries()) {
    const [from, to] = key.split('|');
    interGroupImports.push({ from, to, count });
  }

  // F. Intra-Group Import Density
  const intraGroupDensity = {};
  for (const g of Object.keys(directoryGroups)) {
    let internal = 0;
    let total = 0;
    for (const e of importEdges) {
      const sg = idToGroup[e.source];
      const tg = idToGroup[e.target];
      if (sg === g || tg === g) {
        total++;
        if (sg === g && tg === g) internal++;
      }
    }
    intraGroupDensity[g] = {
      internalEdges: internal,
      totalEdges: total,
      density: total > 0 ? +(internal / total).toFixed(3) : 0
    };
  }

  // G. Directory Pattern Matching
  const dirPatterns = {
    api: ['routes', 'api', 'controllers', 'endpoints', 'handlers', 'controller', 'routers', 'blueprints', 'serializers'],
    service: ['services', 'core', 'lib', 'domain', 'logic', 'signals', 'internal', 'composables', 'mailers', 'jobs', 'channels'],
    data: ['models', 'db', 'data', 'persistence', 'repository', 'entities', 'migrations', 'sql', 'database', 'schema', 'entity'],
    ui: ['components', 'views', 'pages', 'ui', 'layouts', 'screens'],
    middleware: ['middleware', 'plugins', 'interceptors', 'guards'],
    utility: ['utils', 'helpers', 'common', 'shared', 'tools', 'templatetags', 'pkg'],
    config: ['config', 'constants', 'env', 'settings', 'management', 'commands'],
    test: ['__tests__', 'test', 'tests', 'spec', 'specs'],
    types: ['types', 'interfaces', 'schemas', 'contracts', 'dtos', 'dto', 'request', 'response'],
    hooks: ['hooks'],
    state: ['store', 'state', 'reducers', 'actions', 'slices'],
    assets: ['assets', 'static', 'public'],
    entry: ['cmd', 'bin'],
    documentation: ['docs', 'documentation', 'wiki'],
    'infrastructure': ['deploy', 'deployment', 'infra', 'infrastructure', 'k8s', 'kubernetes', 'helm', 'charts', 'terraform', 'tf', 'docker'],
    'ci-cd': ['.github', '.gitlab', '.circleci']
  };
  const dirPatternLookup = {};
  for (const [label, names] of Object.entries(dirPatterns)) {
    for (const name of names) dirPatternLookup[name] = label;
  }

  const patternMatches = {};
  for (const g of Object.keys(directoryGroups)) {
    const gLower = g.toLowerCase();
    if (dirPatternLookup[gLower]) {
      patternMatches[g] = dirPatternLookup[gLower];
    }
  }

  // H. Deployment Topology Detection
  const infraFiles = [];
  let hasDockerfile = false, hasCompose = false, hasK8s = false, hasTerraform = false, hasCI = false;
  for (const n of fileNodes) {
    const fp = n.filePath || '';
    const base = path.basename(fp);
    if (/^Dockerfile/i.test(base)) { hasDockerfile = true; infraFiles.push(fp); }
    if (/docker-compose/i.test(base)) { hasCompose = true; infraFiles.push(fp); }
    if (/\.tf$|\.tfvars$/.test(base)) { hasTerraform = true; infraFiles.push(fp); }
    if (/k8s|kubernetes|helm/i.test(fp)) { hasK8s = true; infraFiles.push(fp); }
    if (/^\.github\/workflows\//.test(fp) || /\.gitlab-ci\.yml$/.test(base) || base === 'Jenkinsfile') { hasCI = true; infraFiles.push(fp); }
    if (base === 'Makefile') { infraFiles.push(fp); }
  }

  const deploymentTopology = {
    hasDockerfile, hasCompose, hasK8s, hasTerraform, hasCI,
    infraFiles: Array.from(new Set(infraFiles))
  };

  // I. Data Pipeline Detection
  const schemaFiles = [];
  const migrationFiles = [];
  const dataModelFiles = [];
  const apiHandlerFiles = [];
  for (const n of fileNodes) {
    const fp = n.filePath || '';
    const g = idToGroup[n.id];
    if (/\.sql$/.test(fp) || /\.graphql$|\.gql$|\.proto$/.test(fp)) schemaFiles.push(fp);
    if (/migrations\//.test(fp)) migrationFiles.push(fp);
    if (patternMatches[g] === 'data') dataModelFiles.push(fp);
    if (patternMatches[g] === 'api') apiHandlerFiles.push(fp);
  }

  const dataPipeline = { schemaFiles, migrationFiles, dataModelFiles, apiHandlerFiles };

  // J. Documentation Coverage
  const docFiles = fileNodes.filter(n => n.type === 'document' || /\.md$|\.rst$/i.test(n.filePath || ''));
  const groupsWithDocsSet = new Set();
  // Simplistic: a group "has docs" if there's a README in that dir, or type document node referencing files in group (we don't have per-file doc-target edges here, so just check dir README)
  for (const n of docFiles) {
    const fp = n.filePath || '';
    const g = groupFor(fp);
    groupsWithDocsSet.add(g);
  }
  const totalGroups = Object.keys(directoryGroups).length;
  const groupsWithDocs = Array.from(groupsWithDocsSet).filter(g => directoryGroups[g]).length;
  const undocumentedGroups = Object.keys(directoryGroups).filter(g => !groupsWithDocsSet.has(g));
  const docCoverage = {
    groupsWithDocs,
    totalGroups,
    coverageRatio: totalGroups > 0 ? +(groupsWithDocs / totalGroups).toFixed(3) : 0,
    undocumentedGroups
  };

  // K. Dependency Direction
  const dependencyDirection = [];
  const seenPairs = new Set();
  for (const { from, to, count } of interGroupImports) {
    const pairKey = [from, to].sort().join('|');
    if (seenPairs.has(pairKey)) continue;
    seenPairs.add(pairKey);
    const reverse = interGroupMap.get(to + '|' + from) || 0;
    if (count > reverse) {
      dependencyDirection.push({ dependent: from, dependsOn: to });
    } else if (reverse > count) {
      dependencyDirection.push({ dependent: to, dependsOn: from });
    }
  }

  // fileStats
  const filesPerGroup = {};
  for (const [g, ids] of Object.entries(directoryGroups)) filesPerGroup[g] = ids.length;
  const nodeTypeCounts = {};
  for (const [t, ids] of Object.entries(nodeTypeGroups)) nodeTypeCounts[t] = ids.length;

  const result = {
    scriptCompleted: true,
    directoryGroups,
    nodeTypeGroups,
    crossCategoryEdges,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    fileStats: {
      totalFileNodes: fileNodes.length,
      filesPerGroup,
      nodeTypeCounts
    },
    fileFanIn,
    fileFanOut
  };

  try {
    fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  } catch (e) {
    console.error('Failed to write output JSON: ' + e.message);
    process.exit(1);
  }

  process.exit(0);
}

main();
