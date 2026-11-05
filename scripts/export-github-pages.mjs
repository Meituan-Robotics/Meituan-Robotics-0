import { cp, mkdir, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const projectDir = process.cwd();
const clientDir = path.join(projectDir, "dist", "client");
const outputDir = path.join(projectDir, ".github-pages");
const exportUrl = (process.env.EXPORT_URL ?? "http://127.0.0.1:3001").replace(/\/$/, "");
const repository = process.env.GITHUB_REPOSITORY?.split("/").at(-1) ?? "Robotic-0";
const owner = process.env.GITHUB_REPOSITORY_OWNER ?? "YOUR-USERNAME";
const basePath = process.env.GITHUB_PAGES_BASE_PATH ?? `/${repository}`;
const publicOrigin = process.env.GITHUB_PAGES_ORIGIN ?? `https://${owner.toLowerCase()}.github.io`;

const response = await fetch(`${exportUrl}${basePath}/`);
if (!response.ok) {
  throw new Error(`Unable to render the project page: ${response.status} ${response.statusText}`);
}

let html = await response.text();

// The production render contains browser-ready React/Vite assets, but absolute
// URLs point to the domain root. GitHub project sites live under /<repository>/,
// so rewrite both visible markup and the embedded RSC payload consistently.
const escapedBasePath = basePath.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
for (const directory of ["_next", "assets", "video"]) {
  html = html.replace(
    new RegExp(`(?<!${escapedBasePath})/${directory}/`, "g"),
    `${basePath}/${directory}/`,
  );
}

for (const file of ["Meituan-Robotics-0.pdf", "favicon.png", "og.png"]) {
  html = html.replace(
    new RegExp(`(?<!${escapedBasePath})/${file.replaceAll(".", "\\.")}`, "g"),
    `${basePath}/${file}`,
  );
}

html = html
  .replace(
    new RegExp(
      `https?://(?:127\\.0\\.0\\.1|localhost)(?::\\d+)?${escapedBasePath}/og\\.png`,
      "g",
    ),
    `${publicOrigin}${basePath}/og.png`,
  )
  .replace('nav:{"pathname":"/"', `nav:{"pathname":"${basePath}/"`);

// GitHub Pages cannot serve Vinext's RSC navigation endpoint. Keep the complete
// server-rendered page, remove the RSC bootstrap, and attach a small standalone
// enhancement script for the page's interactive controls.
const mainEndMarker = "</main>";
const mainEnd = html.lastIndexOf(mainEndMarker);
if (mainEnd < 0) throw new Error("Unable to find the rendered project page root");

html = `${html.slice(0, mainEnd + mainEndMarker.length)}</body></html>`
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "")
  .replace(/<link\b[^>]*\brel=["']modulepreload["'][^>]*\/?\s*>/gi, "");

const publicPageUrl = `${publicOrigin}${basePath}/`;
const publicImageUrl = `${publicOrigin}${basePath}/og.png`;
const staticMetadata = [
  "<title>Meituan-Robotics-0</title>",
  '<meta name="description" content="A Vision-Language-Action foundation model for precise bimanual desktop manipulation.">',
  '<meta name="robots" content="noindex, nofollow, nocache">',
  '<meta name="googlebot" content="noindex, nofollow, noimageindex">',
  '<meta property="og:title" content="Meituan-Robotics-0">',
  '<meta property="og:description" content="Vision-Language-Action for Desktop Manipulation">',
  '<meta property="og:type" content="website">',
  `<meta property="og:url" content="${publicPageUrl}">`,
  `<meta property="og:image" content="${publicImageUrl}">`,
  '<meta property="og:image:width" content="1672">',
  '<meta property="og:image:height" content="941">',
  '<meta property="og:image:alt" content="Meituan-Robotics-0">',
  '<meta name="twitter:card" content="summary_large_image">',
  '<meta name="twitter:title" content="Meituan-Robotics-0">',
  '<meta name="twitter:description" content="Vision-Language-Action for Desktop Manipulation">',
  `<meta name="twitter:image" content="${publicImageUrl}">`,
  `<link rel="canonical" href="${publicPageUrl}">`,
  `<link rel="icon" href="${basePath}/favicon.png">`,
].join("");

html = html
  .replace("</head>", `${staticMetadata}</head>`)
  .replace(
    "</body>",
    `<script src="${basePath}/github-pages.js" defer></script><script src="${basePath}/figure-viewer.js" defer></script></body>`,
  );

if (/vinext\.navigationRuntime|type=["']module["']|rel=["']modulepreload["']/.test(html)) {
  throw new Error("Vinext runtime markup remained in the static GitHub Pages export");
}

const validationHtml = html
  .replaceAll(`${basePath}/_next/`, "")
  .replaceAll(`${basePath}/assets/`, "")
  .replaceAll(`${basePath}/video/`, "");
const unresolvedAsset = validationHtml.match(/\/(?:_next|assets|video)\//);
if (unresolvedAsset) {
  throw new Error(`Found an unscoped asset URL after export: ${unresolvedAsset[0]}`);
}

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });

const prefixedDirectoryName = basePath.replace(/^\//, "");
for (const entry of await readdir(clientDir, { withFileTypes: true })) {
  if (entry.name === prefixedDirectoryName) continue;
  await cp(path.join(clientDir, entry.name), path.join(outputDir, entry.name), {
    recursive: true,
  });
}

const prefixedClientDir = path.join(clientDir, prefixedDirectoryName);
for (const entry of await readdir(prefixedClientDir, { withFileTypes: true })) {
  await cp(path.join(prefixedClientDir, entry.name), path.join(outputDir, entry.name), {
    recursive: true,
  });
}
await Promise.all([
  writeFile(path.join(outputDir, "index.html"), html),
  writeFile(path.join(outputDir, "404.html"), html),
  writeFile(path.join(outputDir, ".nojekyll"), ""),
]);

console.log(`GitHub Pages export created at ${outputDir}`);
console.log(`Expected public URL: ${publicOrigin}${basePath}/`);
