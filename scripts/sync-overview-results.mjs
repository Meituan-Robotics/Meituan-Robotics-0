import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

// The September artwork is archived; the live Overview is now a direct PPT export.
const dataUrl = new URL("../data/archived-overview-results.json", import.meta.url);
const overviewUrl = new URL("../public/assets/main-overview-20260902.svg", import.meta.url);
export const startMarker = "  <!-- BEGIN GENERATED EVALUATION: data/project-results.json -->";
export const endMarker = "  <!-- END GENERATED EVALUATION -->";
const legacyStartMarker = "  <!-- Real-robot averages reported in Figure 10; HG-DAgger comparison. -->";

const baselineColor = "#969696";
const highlightColor = "#ffbd00";
const plotBottom = 1008;
const plotHeight = 85;
const n = (value, digits = 4) => Number(value.toFixed(digits));
const escapeXml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;",
})[character]);

function assertNumber(value, min, max, name) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`Invalid ${name}: expected a finite number between ${min} and ${max}`);
  }
}

export function validateResults(results) {
  for (const [key, count] of [["robotwin", 4], ["robodojo", results.robodojo?.models?.length]]) {
    const chart = results[key];
    if (!chart || !count || count < 2 || chart.models?.length !== count) throw new Error(`Expected ${count} ${key} entries`);
    if (chart.models.filter((model) => model.highlight).length !== 1) throw new Error(`Expected one highlighted ${key} entry`);
    if (new Set(chart.models.map((model) => model.label)).size !== count) throw new Error(`Duplicate ${key} labels`);
    for (const model of chart.models) {
      if (typeof model.label !== "string" || !model.label.trim()) throw new Error(`Missing ${key} model label`);
      assertNumber(model.value, chart.min, chart.max, `${key} ${model.label}`);
    }
  }
  for (const key of ["realRobot", "ethernetRollback"]) {
    for (const method of ["baseline", "mr0"]) {
      for (const metric of ["sr", "ps"]) assertNumber(results[key][method][metric], 0, 1, `${key} ${method} ${metric}`);
    }
  }
  for (const key of ["letterBlock", "garment"]) {
    for (const metric of ["sr", "ps"]) assertNumber(results.demos[key][metric], 0, 1, `${key} ${metric}`);
  }
  if (results.demos.ethernet.resultRef !== "ethernetRollback.mr0") throw new Error("Ethernet demo must reference its CFG-DAgger experiment result");
}

function bar({ label, value, x, width, min = 0, max, highlight = false, metric = "score" }) {
  const height = n((value - min) / (max - min) * plotHeight);
  const y = n(plotBottom - height);
  return `    <rect data-model="${escapeXml(label)}" data-${metric}="${value}" x="${x}" y="${y}" width="${width}" height="${height}" fill="${highlight ? highlightColor : baselineColor}"/>`;
}

function modelLabel(label, x) {
  const lines = label === "InternVLA-A1.5" ? ["InternVLA-", "A1.5"] : label === "Spatial Forcing" ? ["Spatial", "Forcing"] : [label];
  return `<text x="${x}" y="1018">${escapeXml(lines[0])}${lines[1] ? `<tspan x="${x}" dy="7">${escapeXml(lines[1])}</tspan>` : ""}</text>`;
}

function sourceLabel(label) {
  return escapeXml(label.replace(/^Figure /, "Fig. "));
}

export function renderOverviewResults(results) {
  validateResults(results);
  const real = results.realRobot;
  const realBars = [
    { model: real.baseline, metric: "sr", x: 1035 },
    { model: real.mr0, metric: "sr", x: 1057, highlight: true },
    { model: real.baseline, metric: "ps", x: 1105 },
    { model: real.mr0, metric: "ps", x: 1127, highlight: true },
  ];
  const twin = results.robotwin;
  const dojo = results.robodojo;
  return `${startMarker}
  <!-- Real-robot averages reported in ${escapeXml(real.source)}; ${escapeXml(real.protocol)} comparison. -->
  <rect x="994" y="906" width="177" height="129" fill="#fff"/>
  <g id="real-robot-comparison" font-size="6" fill="#555">
    <rect x="1049" y="910" width="6" height="5" fill="${baselineColor}"/>
    <text x="1058" y="915">${escapeXml(real.baseline.label)}</text>
    <rect x="1091" y="910" width="6" height="5" fill="${highlightColor}"/>
    <text x="1100" y="915">${escapeXml(real.mr0.label)}</text>
    <path d="M1016 923V1008H1164" stroke="#bfbfbf" stroke-width=".6" fill="none"/>
    <path d="M1016 923H1164M1016 965.5H1164" stroke="#e7e7e7" stroke-width=".4" fill="none"/>
    <text x="1011" y="926" text-anchor="end">1.0</text><text x="1011" y="968" text-anchor="end">0.5</text><text x="1011" y="1010" text-anchor="end">0</text>
${realBars.map(({ model, metric, x, highlight }) => bar({ label: model.label, value: model[metric], x, width: 17, max: 1, highlight, metric })).join("\n")}
    <g text-anchor="middle" font-size="7" fill="#333">${realBars.map(({ model, metric, x }) => `<text x="${x + 8.5}" y="${Math.round(plotBottom - model[metric] * plotHeight - 3.5)}">${model[metric].toFixed(2)}</text>`).join("")}</g>
    <text x="1054" y="1018" text-anchor="middle" font-size="7">Avg. SR</text><text x="1124" y="1018" text-anchor="middle" font-size="7">Avg. PS</text>
    <text x="1090" y="1030" text-anchor="middle">Matched ${escapeXml(real.protocol)}</text>
  </g>
  <!-- RoboTwin 2.0 overall SR, ${escapeXml(twin.source)}. -->
  <rect x="1181" y="909" width="187" height="128" fill="#fff"/>
  <g id="robotwin-comparison" font-size="6" fill="#555">
    <path d="M1195 923V1008H1363" stroke="#bfbfbf" stroke-width=".6" fill="none"/>
    <path d="M1195 923H1363M1195 951.333H1363M1195 979.667H1363" stroke="#e7e7e7" stroke-width=".4" fill="none"/>
    <text x="1191" y="926" text-anchor="end">95</text><text x="1191" y="954" text-anchor="end">90</text><text x="1191" y="982" text-anchor="end">85</text><text x="1191" y="1010" text-anchor="end">80</text>
${twin.models.map((model, index) => bar({ ...model, x: 1208 + index * 42, width: 19, min: twin.min, max: twin.max, metric: "sr" })).join("\n")}
    <g text-anchor="middle" font-size="7" fill="#333">${twin.models.map((model, index) => `<text x="${1217.5 + index * 42}" y="${Math.round(plotBottom - (model.value - twin.min) / (twin.max - twin.min) * plotHeight - 3.5)}">${model.value.toFixed(1)}</text>`).join("")}</g>
    <g text-anchor="middle">${twin.models.map((model, index) => modelLabel(model.label, 1217.5 + index * 42)).join("")}</g>
    <text x="1279" y="1034" text-anchor="middle">Overall SR (%) · scale ${twin.min}–${twin.max}</text>
  </g>
  <!-- RoboDojo baselines from ${escapeXml(dojo.sources.baselines)}; MR0 from ${escapeXml(dojo.sources.mr0)}. -->
  <rect x="1370" y="909" width="204" height="128" fill="#fff"/>
  <g id="robodojo-comparison" font-size="6" fill="#555">
    <path d="M1390 923V1008H1570" stroke="#bfbfbf" stroke-width=".6" fill="none"/>
    <path d="M1390 923H1570M1390 965.5H1570" stroke="#e7e7e7" stroke-width=".4" fill="none"/>
    <text x="1386" y="926" text-anchor="end">20</text><text x="1386" y="968" text-anchor="end">10</text><text x="1386" y="1010" text-anchor="end">0</text>
${dojo.models.map((model, index) => bar({ ...model, x: 1399 + index * 36, width: 18, max: 20 })).join("\n")}
    <g text-anchor="middle" font-size="7" fill="#333">${dojo.models.map((model, index) => `<text x="${1408 + index * 36}" y="${n(plotBottom - model.value / 20 * plotHeight - 3.5, 1)}">${model.value.toFixed(2)}</text>`).join("")}</g>
    <g text-anchor="middle" font-size="5.8">${dojo.models.map((model, index) => modelLabel(model.label, 1408 + index * 36)).join("")}</g>
    <text x="1480" y="1034" text-anchor="middle">${sourceLabel(dojo.sources.baselines)} baselines · MR0: ${sourceLabel(dojo.sources.mr0)}</text>
  </g>
${endMarker}`;
}

export function splitOverview(svg) {
  let start = svg.indexOf(startMarker);
  let end;
  if (start !== -1) {
    const endIndex = svg.indexOf(endMarker, start);
    if (endIndex === -1 || svg.indexOf(startMarker, start + 1) !== -1 || svg.indexOf(endMarker) !== endIndex || svg.indexOf(endMarker, endIndex + 1) !== -1) throw new Error("Invalid generated results markers in overview SVG");
    end = endIndex + endMarker.length;
  } else {
    start = svg.indexOf(legacyStartMarker);
    end = svg.lastIndexOf("\n</g>\n</svg>");
    if (start === -1 || end <= start) throw new Error("Unable to locate the overview result mini-charts; no file was changed");
  }
  return { prefix: svg.slice(0, start), generated: svg.slice(start, end), suffix: svg.slice(end) };
}

export function synchronizeOverview(svg, results) {
  const { prefix, suffix } = splitOverview(svg);
  return prefix + renderOverviewResults(results) + suffix;
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length > 1 || (args.length === 1 && !["--check", "--write"].includes(args[0]))) {
    throw new Error("Usage: node scripts/sync-overview-results.mjs [--check | --write]");
  }
  const [json, svg] = await Promise.all([readFile(dataUrl, "utf8"), readFile(overviewUrl, "utf8")]);
  const updated = synchronizeOverview(svg, JSON.parse(json));
  if (updated === svg) {
    console.log("Archived overview charts match data/archived-overview-results.json.");
    return;
  }
  if (args[0] !== "--write") throw new Error("Overview results are out of sync. Run node scripts/sync-overview-results.mjs --write and review the SVG diff.");
  await writeFile(overviewUrl, updated);
  console.log(`Updated only the overview result charts: ${fileURLToPath(overviewUrl)}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
