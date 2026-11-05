import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const basePath = process.env.GITHUB_PAGES_BASE_PATH ?? "";

  return worker.fetch(
    new Request(`http://localhost${basePath}/`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the Meituan-Robotics-0 project page", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  const readableHtml = html.replace(/<!--[\s\S]*?-->/g, "");
  assert.match(readableHtml, /reaches 93\.6% success/);
  assert.match(readableHtml, /an average success rate of 14\.63% on RoboDojo, as reported in Table 5\./);
  assert.match(readableHtml, /Table 5\. Real-world/);
  assert.equal((html.match(/data-figure-zoom="true"|data-figure-zoom=""/g) ?? []).length, 3);
  assert.match(html, /aria-label="View System overview at full size"/);
  assert.match(html, /aria-label="View Vision-language data at full size"/);
  assert.match(html, /aria-label="View Event-driven agent at full size"/);
  assert.match(html, /<title>Meituan-Robotics-0<\/title>/i);
  assert.doesNotMatch(html, /LongCat Robotics Team/i);
  assert.match(html, /main-overview-20261103\.svg\?v=20261103-paper/);
  assert.doesNotMatch(html, /system-overview|Rollback and CFG-DAgger/);
  assert.match(html, />Overview</);
  assert.match(html, />Data</);
  assert.match(html, />Method</);
  assert.match(html, />Evaluation</);
  assert.match(html, />Real-World Demos</);
  assert.match(html, />Conclusion</);
  assert.match(html, />Citation</);
  assert.match(html, /name="robots" content="noindex, nofollow, nocache"/);
  assert.match(html, /name="googlebot" content="noindex, nofollow, noimageindex"/);
  assert.doesNotMatch(html, />Applications</);
  assert.match(html, /Page reading progress/);
  assert.match(html, /15,060h/);
  assert.ok(
    html.indexOf("main-overview-20261103.svg") <
      html.indexOf("Meituan-Robotics-0 (MR0) is a vision-language-action foundation model"),
  );
  assert.doesNotMatch(html, /Unified Action Representation|Model Architecture|action-flow/);
  assert.match(html, /training-timeline/);
  assert.doesNotMatch(html, /method-facts|stage-grid/);
  assert.match(html, /<strong>14\.63%<\/strong><span>RoboDojo avg\. SR<\/span>/);
  assert.match(html, /RoboTwin 2\.0/);
  assert.match(html, /RoboDojo/);
  assert.match(html, /14\.63/);
  assert.match(html, /Real Robot/);
  assert.match(html, /RECAP/);
  assert.doesNotMatch(html, /real-robot-results\.png|robotwin-results\.png/);
  assert.doesNotMatch(html, /robot-data\.png/);
  assert.doesNotMatch(html, /<table\b/i);
  assert.match(html, /Composition of the 15,060-hour robot dataset/);
  assert.match(html, /Embodiment-wise Duration/);
  assert.match(html, /Select a data source/);
  assert.match(html, /data-source="inhouse"/);
  assert.match(html, /data-source="open-real"/);
  assert.match(html, /data-source="open-sim"/);
  assert.match(html, /Licensed Real/);
  assert.match(html, /RoboDesk/);
  assert.doesNotMatch(html, /data-active-source=/);
  assert.match(html, /AgiBot G1/);
  assert.match(html, /RealMan Aida-L/);
  assert.match(html, /RoboTwin 2\.0 overall success-rate comparison, shown on an 80 to 95 percent scale/);
  assert.match(html, /Average real-robot success rate and progress score under matched HG-DAgger post-training/);
  assert.doesNotMatch(html, /scale 60–100%/);
  assert.match(html, /class="metric-pair" role="group" aria-label="SR 0\.78, PS 0\.91"/);
  assert.doesNotMatch(html, /\+0\.09 SR \/ \+0\.06 PS/);
  assert.match(html, /vertical-chart/);
  assert.match(html, /grouped-vertical-chart/);
  assert.match(html, /class="group-scale"/);
  assert.match(html, /bottom:0%">0\.6/);
  assert.match(html, /bottom:100%">1\.0/);
  assert.doesNotMatch(html, /Agent trace metrics|≈2 Hz|10 events|6 events/);
  assert.doesNotMatch(html, /\+0\.4 pp/);
  assert.match(html, /vertical-scale/);
  assert.doesNotMatch(html, /\+0\.10 SR \/ \+0\.23 PS/);
  assert.doesNotMatch(html, /0\.08 → 0\.86|15\.27|>Score</);
  assert.doesNotMatch(html, /Capability Breakdown|RoboTwin clean-scene|RoboTwin randomized-scene/);
  assert.doesNotMatch(html, /Data Curation|data-curation\.png/);
  assert.match(html, /vision-language-data-20261030\.svg/);
  assert.doesNotMatch(html, /action-head\.svg/);
  assert.match(html, /embodied-agent-20261029\.svg/);
  assert.doesNotMatch(html, /rollback-20261029\.svg|vision-language-data-20261029\.svg/);
  assert.equal((html.match(/<img\b/g) ?? []).length, 3);
  assert.match(html, /aria-label="Scrollable event-driven agent architecture" tabindex="0"/);
  assert.doesNotMatch(html, /main-overview\.png|vision-language-data\.png|action-head\.png|embodied-agent\.png/);
  assert.match(html, /event-retrieval-first-frame\.jpg/);
  assert.match(html, /waiting-recovery-first-frame\.jpg/);
  assert.doesNotMatch(html, /event-memory\.png|waiting-recovery\.png/);
  assert.match(html, /10,765 hours/);
  assert.match(html, /Ethernet Insertion/);
  assert.match(html, /block-placement-full\.mp4/);
  assert.match(html, /letter-block-placement\.mp4/);
  assert.match(html, /garment-folding-demo\.mp4/);
  assert.match(html, /network-cable-insertion\.mp4/);
  assert.doesNotMatch(html, /Arrange seven letter blocks in order inside a continuous groove/);
  assert.doesNotMatch(html, /Coordinate two arms while handling deformable cloth/);
  assert.doesNotMatch(html, /Localize, hand over, align, and insert a thin deformable connector/);
  assert.match(html, /block-placement-first-frame\.jpg/);
  assert.match(html, /letter-block-placement-first-frame\.jpg/);
  assert.match(html, /garment-folding-first-frame\.jpg/);
  assert.match(html, /network-cable-insertion-first-frame\.jpg/);
  assert.doesNotMatch(html, /cable-grasp-handover\.mp4|cable-align-insert\.mp4|cable-adaptive-tracking\.mp4|Demo coming soon/);
  assert.match(html, /vla_demo_01_en\.mp4/);
  assert.match(html, /vla_demo_02_en\.mp4/);
  assert.equal((html.match(/\?v=20260828-hq/g) ?? []).length, 8);
  assert.equal((html.match(/\?v=20260828-block1440/g) ?? []).length, 2);
  assert.equal((html.match(/\?v=20260828-letter1440/g) ?? []).length, 4);
  assert.doesNotMatch(html, /Each video can be viewed at 1×, 2×, or 3× playback speed\./);
  assert.doesNotMatch(html, /Code, model weights, dataset, and demos: coming soon\./);
  assert.match(html, />Copy Citation</);
  assert.match(html, /<option value="1"/);
  assert.match(html, /<option value="2"/);
  assert.match(html, /<option value="3"/);
  assert.doesNotMatch(html, /<option value="(?:0\.5|1\.5)"/);
  assert.match(html, /Cross-task Event Retrieval demo/);
  assert.match(html, /Subtask Waiting and Recovery demo/);
  assert.doesNotMatch(html, /Demo 04 · Coming soon|Demo 05 · Coming soon|demo placeholder/);
  assert.match(html, /og:image/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);

  // Keep the original page structure while auditing corrected paper results.
  assert.doesNotMatch(html, /14\.95|9\.53|0\.94|ranks? fourth|first place|SOTA Performance/i);
  assert.doesNotMatch(html, /without benchmark-specific enhancements|outcome-labeled autonomous rollouts/);
  const resultCards = [...html.matchAll(/<article class="result-card">([\s\S]*?)<\/article>/g)].map((match) => match[1]);
  assert.equal(resultCards.length, 4);
  for (const card of resultCards) {
    assert.doesNotMatch(card, /<footer\b/);
    assert.match(card, /<header\b/);
    assert.match(card, /vertical-chart/);
  }
  const roboDojoCard = resultCards.find((card) => /<h4>RoboDojo<\/h4>/.test(card));
  assert.ok(roboDojoCard, "Missing RoboDojo model-comparison card");
  for (const value of ["Hy-Embodied-0.5-VLA", "8.80", "MR0", "14.63", "Xiaomi-Robotics-1", "13.93", "GalaxeaVLA (G0.5)", "14.88", "DM0.5", "19.34", "Rex-M1 Preview", "27.44", "Simate-beta", "27.96"]) {
    assert.ok(roboDojoCard.includes(value), `Missing selected RoboDojo model or Score ${value}`);
  }
  assert.doesNotMatch(roboDojoCard, /Selected methods · scale/);
  assert.match(roboDojoCard, /RoboDojo average success rate \(%\) comparison from Table 5, retaining the selected methods from Figure 1; not a complete leaderboard/);
  assert.equal((roboDojoCard.match(/class="vertical-column/g) ?? []).length, 7);
  assert.doesNotMatch(roboDojoCard, /scale 0–20/);
  assert.match(roboDojoCard, /vertical-column highlight/);
  assert.doesNotMatch(roboDojoCard, /VLACT|InternVLA|Spatial Forcing|SimpleMem|KinRT/);
  assert.match(roboDojoCard, /Average SR \(%\)/);
  assert.doesNotMatch(roboDojoCard, /Average Score|19\.68/);
  assert.match(html, /The RoboDojo comparison retains the selected baselines from Figure 1 and uses average success rates from Table 5\./);
  assert.doesNotMatch(html, /initial-scene reference and latent-action history|without the embodied-agent system|using compact historical context/);
  assert.match(html, /CFG-DAgger compared with RECAP on Ethernet insertion using the same demonstrations and Rollback episodes/);
  assert.match(html, /RECAP: 0\.60/);
  assert.match(html, /RECAP: 0\.67/);
  assert.match(html, /CFG-DAgger: 0\.70/);
  assert.match(html, /CFG-DAgger: 0\.90/);
  assert.match(html, /class="metric-pair" role="group" aria-label="SR 0\.70, PS 0\.90"/);
  const demoCards = [...html.matchAll(/<article class="task-card">([\s\S]*?)<\/article>/g)].map((match) => match[1]);
  assert.equal(demoCards.length, 3);
  const expectedDemos = [["Letter-block Placement", "0.83", "0.96"], ["Garment Folding", "0.83", "0.93"], ["Ethernet Insertion", "0.70", "0.90"]];
  demoCards.forEach((card, index) => {
    const [title, sr, ps] = expectedDemos[index];
    assert.ok(card.includes(`<h4>${title}</h4>`));
    assert.ok(card.includes(`<dt>SR</dt><dd>${sr}</dd>`));
    assert.ok(card.includes(`<dt>PS</dt><dd>${ps}</dd>`));
    assert.match(card, /<video\b/);
    assert.doesNotMatch(card, /task-protocol|HG-DAgger|CFG-DAgger/);
  });
  assert.match(html, /time-based execution efficiency/);
  assert.doesNotMatch(html, /150\.7M VLM samples from 64 sub-datasets/);
  assert.match(html, /Alternative post-training algorithms: HG-DAgger, RECAP, and CFG-DAgger/);
  assert.match(html, /trained anew from the same/);
  assert.match(html, /Meituan-Robotics-0\.pdf\?v=20261103/);
});

test("keeps the finished site free of starter-preview dependencies", async () => {
  const [page, layout, packageJson, styles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(page, /Meituan-Robotics-0/);
  assert.match(layout, /title:\s*"Meituan-Robotics-0"/);
  assert.doesNotMatch(page + layout, /LongCat Robotics Team/i);
  assert.match(layout, /\/og\.png/);
  assert.doesNotMatch(page, /_sites-preview|SkeletonPreview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(styles, /--data-inhouse:\s*#f2c14e/);
  assert.match(styles, /--data-inhouse-active:\s*#e5ad27/);
  assert.match(styles, /--data-real-world:\s*#e9a200/);
  assert.doesNotMatch(styles, /--data-inhouse:\s*#(?:676d70|e9a200)/i);
  assert.match(styles, /--data-open-real-active:\s*#d9b96f/);
  assert.match(styles, /--data-open-sim-active:\s*#579895/);
  assert.doesNotMatch(styles, /--data-highlight/);
  assert.doesNotMatch(styles, /rgba\(255,\s*209,\s*0,\s*\.12\)/);
  assert.match(styles, /\.source-donut\s*\{[^}]*var\(--data-inhouse\)/);
  assert.match(styles, /\.world-donut\s*\{[^}]*var\(--data-real-world\)/);
  assert.equal((styles.match(/--source-active:\s*var\(--data-(?:inhouse|open-real|open-sim)-active\)/g) ?? []).length, 3);
  assert.equal((styles.match(/--source-tint:\s*rgba\(/g) ?? []).length, 3);
  assert.equal((styles.match(/var\(--source-active\)/g) ?? []).length, 4);
  assert.match(styles, /background:\s*var\(--source-tint\)/);
  assert.match(styles, /data-active-source\]\s+\.source-control\s*\{\s*opacity:\s*\.58/);
  assert.match(styles, /data-active-source\]\s+\.stacked-track i\s*\{\s*opacity:\s*\.32/);
  assert.match(styles, /opacity:\s*\.78;\s*stroke:\s*var\(--source-active\)/);
  assert.match(
    styles,
    /top:\s*calc\(var\(--training-axis-y\) - var\(--training-stage-offset\)\)/,
  );
});

function readTrackDimensions(video) {
  const typeOffset = video.indexOf(Buffer.from("tkhd"));
  assert.notEqual(typeOffset, -1, "MP4 is missing a track header");

  const boxStart = typeOffset - 4;
  const version = video[typeOffset + 4];
  const dimensionsOffset = boxStart + (version === 1 ? 96 : 84);
  return [
    video.readUInt32BE(dimensionsOffset) / 65536,
    video.readUInt32BE(dimensionsOffset + 4) / 65536,
  ];
}

test("ships every project demo at high-bitrate source quality in a browser-compatible H.264 container", async () => {
  const videos = await Promise.all([
    readFile(new URL("../public/video/block-placement-full.mp4", import.meta.url)),
    readFile(new URL("../public/video/letter-block-placement.mp4", import.meta.url)),
    readFile(new URL("../public/video/garment-folding-demo.mp4", import.meta.url)),
    readFile(new URL("../public/video/network-cable-insertion.mp4", import.meta.url)),
    readFile(new URL("../public/video/vla_demo_01_en.mp4", import.meta.url)),
    readFile(new URL("../public/video/vla_demo_02_en.mp4", import.meta.url)),
  ]);

  const minimumBytes = [
    90 * 1024 * 1024,
    80 * 1024 * 1024,
    70 * 1024 * 1024,
    40 * 1024 * 1024,
    50 * 1024 * 1024,
    20 * 1024 * 1024,
  ];

  for (const [index, video] of videos.entries()) {
    assert.equal(video.includes(Buffer.from("avc1")), true);
    assert.equal(video.includes(Buffer.from("mp4v")), false);
    assert.equal(video.includes(Buffer.from("soun")), false);
    assert.ok(video.indexOf(Buffer.from("moov")) < video.indexOf(Buffer.from("mdat")));
    assert.ok(video.length >= minimumBytes[index]);
    assert.ok(video.length < 100 * 1024 * 1024);
  }

  assert.deepEqual(videos.map(readTrackDimensions), [
    [2560, 1440],
    [2560, 1440],
    [2560, 1440],
    [1920, 1080],
    [1280, 720],
    [1280, 720],
  ]);
});

test("ships the latest PPT overview and preserves the other figure slots", async () => {
  const figures = await Promise.all([
    readFile(new URL("../public/assets/main-overview-20261103.svg", import.meta.url), "utf8"),
    readFile(new URL("../public/assets/vision-language-data-20261030.svg", import.meta.url), "utf8"),
    readFile(new URL("../public/assets/embodied-agent-20261029.svg", import.meta.url), "utf8"),
  ]);

  for (const figure of figures) {
    const root = figure.match(/<svg\b[^>]*>/)?.[0] ?? "";
    const preserveAspectRatio = root.match(/preserveAspectRatio="([^"]+)"/)?.[1];
    assert.ok(preserveAspectRatio === undefined || preserveAspectRatio === "xMidYMid meet");
    const viewBox = root.match(/viewBox="([^"]+)"/)?.[1].split(/[\s,]+/).map(Number);
    assert.equal(viewBox?.length, 4);
    assert.ok(viewBox.every(Number.isFinite));
    assert.ok(viewBox[2] > 0 && viewBox[3] > 0);
    // PDF artwork can contain both vector paths and original embedded images.
    assert.match(figure, /<(?:path|image|use)\b/);
  }
});

test("preserves the archived generated overview results", async () => {
  const overview = await readFile(new URL("../public/assets/main-overview-20260902.svg", import.meta.url), "utf8");
  const chart = overview.slice(overview.indexOf('<g id="robodojo-comparison"'));
  const bars = [...chart.matchAll(/<rect data-model="([^"]+)" data-score="([^"]+)"[^>]*y="([^"]+)"[^>]*height="([^"]+)"[^>]*fill="([^"]+)"/g)];
  assert.deepEqual(bars.map(([, model, score]) => [model, Number(score)]), [
    ["VLACT", 10.66],
    ["InternVLA-A1.5", 11.15],
    ["π0.5", 11.41],
    ["Spatial Forcing", 12.38],
    ["MR0", 19.68],
  ]);
  for (const [, model, score, y, height, fill] of bars) {
    assert.ok(Math.abs(Number(height) - Number(score) / 20 * 85) < 0.001);
    assert.ok(Math.abs(Number(y) + Number(height) - 1008) < 0.001);
    assert.equal(fill, model === "MR0" ? "#ffbd00" : "#969696");
  }
  assert.match(chart, /Fig\. 1 baselines · MR0: Table 5/);
  assert.doesNotMatch(chart, /SimpleMem|KinRT|Hy-Embodied|14\.95|14\.07|9\.53/);
});

test("ships the exact revised technical report supplied for the November 3 update", async () => {
  const report = await readFile(new URL("../public/Meituan-Robotics-0.pdf", import.meta.url));
  assert.equal(report.subarray(0, 5).toString(), "%PDF-");
  assert.equal(
    createHash("sha256").update(report).digest("hex"),
    "cf689b2f7b7efd06c06095a402d34651c8f3495c4613b2199dab74c2e11efc29",
  );
});
