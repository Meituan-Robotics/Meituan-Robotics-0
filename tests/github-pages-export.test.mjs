import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const output = new URL("../.github-pages/", import.meta.url);

test("exports a self-contained GitHub Pages project site", async () => {
  const rawHtml = await readFile(new URL("index.html", output), "utf8");
  const repository = process.env.GITHUB_REPOSITORY?.split("/").at(-1) ?? "Robotic-0";
  const basePath = process.env.GITHUB_PAGES_BASE_PATH ?? `/${repository}`;
  const owner = process.env.GITHUB_REPOSITORY_OWNER ?? "YOUR-USERNAME";
  const origin = process.env.GITHUB_PAGES_ORIGIN ?? `https://${owner.toLowerCase()}.github.io`;
  // Normalize deployment-specific URLs for the existing content assertions.
  const html = rawHtml.replaceAll(origin, "https://wyatt2021.github.io")
    .replaceAll(`${basePath}/`, "/Robotic-0/");
  assert.ok(rawHtml.includes(`property="og:url" content="${origin}${basePath}/"`));
  const notFoundHtml = await readFile(new URL("404.html", output), "utf8");

  assert.match(html, /<title>Meituan-Robotics-0<\/title>/);
  assert.doesNotMatch(html, /LongCat Robotics Team/i);
  assert.match(html, /name="robots" content="noindex, nofollow, nocache"/);
  assert.match(html, /\/Robotic-0\/_next\/static\/css\//);
  assert.match(html, /\/Robotic-0\/video\/vla_demo_01_en\.mp4/);
  assert.match(html, /\/Robotic-0\/assets\/main-overview-20261103\.svg\?v=20261103-paper/);
  assert.match(html, /\/Robotic-0\/assets\/vision-language-data-20261030\.svg/);
  assert.match(html, /\/Robotic-0\/assets\/embodied-agent-20261029\.svg/);
  assert.doesNotMatch(html, /system-overview|Rollback and CFG-DAgger|rollback-20261029\.svg/);
  assert.match(html, /href="https:\/\/arxiv\.org\/abs\/2611\.08577"/);
  assert.match(html, /<strong>15\.73%<\/strong><span>RoboDojo avg\. SR<\/span>/);
  assert.match(html, /aria-label="SR 0\.70, PS 0\.90"/);
  assert.match(html, /CFG-DAgger/);
  assert.doesNotMatch(html, /14\.95|9\.53|0\.94|ranks? fourth|first place|SOTA Performance/i);
  const resultCards = [...html.matchAll(/<article class="result-card">([\s\S]*?)<\/article>/g)].map((match) => match[1]);
  assert.equal(resultCards.length, 4);
  for (const card of resultCards) assert.doesNotMatch(card, /<footer\b/);
  assert.doesNotMatch(html, /Selected methods · scale|\+0\.4 pp|\+0\.09 SR \/ \+0\.06 PS|\+0\.10 SR \/ \+0\.23 PS/);
  assert.match(html, /Hy-Embodied-0\.5-VLA/);
  assert.match(html, /InternVLA-A1\.5/);
  assert.match(html, /π0\.5/);
  assert.match(html, /Xiaomi-Robotics-1/);
  assert.match(html, /GalaxeaVLA \(G0\.5\)/);
  assert.match(html, /Rex-M1 Preview/);
  assert.match(html, /Simate-beta/);
  assert.match(html, /WALL-OSS-1\.0-lite/);
  assert.doesNotMatch(html, /SimpleMemVLA|KinRT|VLACT|Spatial Forcing/);
  assert.match(html, /<script src="\/Robotic-0\/github-pages\.js" defer><\/script>/);
  assert.match(html, /<script src="\/Robotic-0\/figure-viewer\.js" defer><\/script>/);
  assert.match(
    html,
    /property="og:image" content="https:\/\/wyatt2021\.github\.io\/Robotic-0\/og\.png"/,
  );
  assert.doesNotMatch(html, /https?:\/\/(?:127\.0\.0\.1|localhost)(?::\d+)?\/Robotic-0\/og\.png/);
  assert.doesNotMatch(html, /vinext\.navigationRuntime/);
  assert.doesNotMatch(html, /<script\b[^>]*type=["']module["']/i);
  assert.doesNotMatch(html, /rel=["']modulepreload["']/i);
  assert.doesNotMatch(html, /\/Robotic-0\/Robotic-0\//);
  assert.equal((html.match(/<video\b/g) ?? []).length, 6);
  const demoCards = [...html.matchAll(/<article class="task-card">([\s\S]*?)<\/article>/g)].map((match) => match[1]);
  assert.equal(demoCards.length, 3);
  for (const card of demoCards) assert.doesNotMatch(card, /task-protocol|HG-DAgger|CFG-DAgger/);
  assert.equal((html.match(/<img\b/g) ?? []).length, 3);
  assert.doesNotMatch(notFoundHtml, /vinext\.navigationRuntime|rel=["']modulepreload["']/i);

  const enhancer = await readFile(new URL("github-pages.js", output), "utf8");
  assert.match(enhancer, /playbackRate/);
  assert.match(enhancer, /copy-citation/);
  assert.match(enhancer, /dataset\.activeSource/);

  const withoutScopedAssets = html
    .replaceAll("/Robotic-0/_next/", "")
    .replaceAll("/Robotic-0/assets/", "")
    .replaceAll("/Robotic-0/video/", "");
  assert.doesNotMatch(withoutScopedAssets, /\/(?:_next|assets|video)\//);

  const references = [...html.matchAll(/\b(?:src|href|poster)="([^"]+)"/g)].map(
    (match) => match[1],
  );
  const localFiles = references.filter((reference) => reference.startsWith("/"));
  for (const reference of localFiles) {
    assert.ok(
      reference.startsWith("/Robotic-0/"),
      `Local asset is outside the project base path: ${reference}`,
    );
    const assetPath = decodeURIComponent(reference.slice("/Robotic-0/".length)).split(/[?#]/)[0];
    if (assetPath) await access(new URL(assetPath, output));
  }

  await Promise.all([
    access(new URL("_next/static/chunks/", output)),
    access(new URL("assets/main-overview-20261103.svg", output)),
    access(new URL("assets/vision-language-data-20261030.svg", output)),
    access(new URL("assets/embodied-agent-20261029.svg", output)),
    access(new URL("video/block-placement-full.mp4", output)),
    access(new URL("video/vla_demo_01_en.mp4", output)),
    access(new URL("github-pages.js", output)),
    access(new URL("figure-viewer.js", output)),
    access(new URL("robots.txt", output)),
    access(new URL(".nojekyll", output)),
  ]);
});
