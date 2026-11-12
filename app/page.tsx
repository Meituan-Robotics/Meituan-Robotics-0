import CopyCitation from "./copy-citation";
import ProjectNav from "./project-nav";
import RevealOnView from "./reveal-on-view";
import RobotDataFigure from "./robot-data-figure";
import VideoPlayer from "./video-player";
import FigureViewer from "./figure-viewer";
import results from "../data/project-results.json";

type BarItem = {
  label: string;
  shortLabel?: string;
  value: number;
  display?: string;
  highlight?: boolean;
};

type VerticalGroup = {
  label: string;
  bars: BarItem[];
};

const citationText = `Meituan Robotics Team. “Meituan-Robotics-0: A Vision-Language-Action
Foundation Model for Desktop Manipulation.” arXiv:2611.08577, 2026.
https://arxiv.org/abs/2611.08577`;

const robotwinMR0 = results.robotwin.models.find((model) => model.highlight)!;
const robodojoMR0 = results.robodojo.models.find((model) => model.highlight)!;
const number = (value: number) => value.toLocaleString("en-US");
const fixed = (value: number) => value.toFixed(2);

function MetricPair({ sr, ps }: { sr: number; ps: number }) {
  return (
    <div className="metric-pair" role="group" aria-label={`SR ${fixed(sr)}, PS ${fixed(ps)}`}>
      <div><span>SR</span><strong>{fixed(sr)}</strong></div>
      <div><span>PS</span><strong>{fixed(ps)}</strong></div>
    </div>
  );
}

function VerticalBarChart({
  items,
  min = 0,
  max,
  label,
  ticks,
  barTopValues = false,
}: {
  items: BarItem[];
  min?: number;
  max: number;
  label: string;
  ticks?: number[];
  barTopValues?: boolean;
}) {
  return (
    <div className={`vertical-chart${ticks ? " has-scale" : ""}${items.length > 5 ? " dense-chart" : ""}${barTopValues ? " bar-top-values" : ""}`} role="group" aria-label={label}>
      {ticks && <div className="vertical-scale" aria-hidden="true">
        {ticks.map((tick) => <span key={tick} style={{ bottom: `${(tick - min) / (max - min) * 100}%` }}><i>{tick}</i></span>)}
      </div>}
      {items.map((item, index) => {
        const height = `${Math.max(0, Math.min(100, (item.value - min) / (max - min) * 100))}%`;
        const value = item.display ?? item.value;
        return (
          <div className={`vertical-column${item.highlight ? " highlight" : ""}`} key={item.label}>
            {!barTopValues && <b>{value}</b>}
            <div className="vertical-track">
              <span style={{ height, transitionDelay: `${index * 65}ms` }} />
              {barTopValues && <b className="vertical-bar-value" style={{ bottom: `calc(${height} + 5px)` }}>{value}</b>}
            </div>
            <span title={item.label}>{item.shortLabel ?? (item.label === "InternVLA-A1.5" ? <>InternVLA-<wbr />A1.5</> : item.label)}</span>
          </div>
        );
      })}
    </div>
  );
}

function GroupedVerticalChart({
  groups,
  min = 0,
  max,
  series,
  label,
  ticks,
}: {
  groups: VerticalGroup[];
  min?: number;
  max: number;
  series: [string, string];
  label: string;
  ticks?: number[];
}) {
  return (
    <div className="grouped-vertical-chart" role="group" aria-label={label}>
      <div className="vertical-legend"><span>{series[0]}</span><span>{series[1]}</span></div>
      <div className={`vertical-groups${ticks ? " has-group-scale" : ""}`}>
        {ticks && <div className="group-scale" aria-hidden="true">
          {ticks.map((tick) => <span key={tick} style={{ bottom: `${(tick - min) / (max - min) * 100}%` }}>{tick.toFixed(1)}</span>)}
        </div>}
        {groups.map((group, groupIndex) => (
          <div className="vertical-group" key={group.label}>
            <div className="group-bars">
              {group.bars.map((bar, barIndex) => (
                <div className={`grouped-column${bar.highlight ? " highlight" : ""}`} key={bar.label} aria-label={`${bar.label}: ${bar.display ?? bar.value}`}>
                  <b>{bar.display ?? bar.value}</b>
                  <div className="vertical-track"><span style={{ height: `${Math.max(0, Math.min(100, (bar.value - min) / (max - min) * 100))}%`, transitionDelay: `${(groupIndex * 2 + barIndex) * 80}ms` }} /></div>
                </div>
              ))}
            </div>
            <strong>{group.label}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <main id="top">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Meituan Robotics home">
          <span className="brand-mark">美团</span>
          <span>Meituan Robotics</span>
        </a>
        <ProjectNav />
      </header>

      <section className="hero section-shell">
        <h1>Meituan-Robotics-0</h1>
        <p className="hero-subtitle">
          A Vision-Language-Action Foundation Model for Desktop Manipulation
        </p>
        <div className="resource-row">
          <a className="resource primary" href="https://arxiv.org/abs/2611.08577">
            Technical Report <span aria-hidden="true">↗</span>
          </a>
          <a className="resource" href="https://github.com/Meituan-Robotics/Meituan-Robotics-0">
            Code <span aria-hidden="true">↗</span>
          </a>
          <a className="resource" href="https://huggingface.co/collections/MeituanRobotics/meituan-robotics-0-6aec7de4477b26c5c73b5e2d">
            Model <span aria-hidden="true">↗</span>
          </a>
          <a className="resource" href="https://huggingface.co/datasets/MeituanRobotics/RoboDesk">
            Dataset <span aria-hidden="true">↗</span>
          </a>
        </div>

        <figure className="hero-demo">
          <VideoPlayer
            src="/video/block-placement-full.mp4?v=20260828-block1440"
            poster="/assets/block-placement-first-frame.jpg"
            ariaLabel="Full Letter-block Placement featured demo"
          />
        </figure>
      </section>

      <section className="project-section section-shell" id="overview">
        <div className="section-heading">
          <h2>Overview</h2>
        </div>
        {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- This diagram scrolls horizontally on smaller screens. */}
        <figure
          className="main-figure overview-main-figure"
          role="region"
          aria-label="Scrollable system overview"
          tabIndex={0}
        >
          <img
            src="/assets/main-overview-20261103.svg?v=robodojo-walloss"
            alt="Overview of the data, training, model, agent, and real-world capabilities of Meituan-Robotics-0"
            width={1771}
            height={1161}
            loading="lazy"
          />
        </figure>
        <FigureViewer src="/assets/main-overview-20261103.svg?v=robodojo-walloss" title="System overview" />
        <div className="overview-copy">
          <p>
            Meituan-Robotics-0 (MR0) is a vision-language-action foundation model
            and embodied-agent system for stationary-base desktop manipulation.
            It combines {results.data.visionLanguageSamples / 1_000_000} million vision-language samples with {number(results.data.robotHours)} hours of
            robot trajectories, then transfers these priors to continuous control
            through a three-stage training recipe. {`The standalone policy reaches ${robotwinMR0.value}% success on RoboTwin 2.0. It achieves an average success rate of ${fixed(robodojoMR0.value)}% on RoboDojo. `}
            Real-world post-training uses demonstrations and human corrections;
            Rollback pairs replaced policy segments with corrective continuations
            for CFG-DAgger. The event-driven agent adds persistent task memory and
            condition-aware resumption for long-horizon execution.
          </p>
        </div>
        <div className="fact-grid" aria-label="Key results">
          <div><strong>{`${results.data.visionLanguageSamples / 1_000_000}M`}</strong><span>vision-language samples</span></div>
          <div><strong>{`${number(results.data.robotHours)}h`}</strong><span>robot trajectory data</span></div>
          <div><strong>{`${robotwinMR0.value}%`}</strong><span>RoboTwin 2.0 overall SR</span></div>
          <div><strong>{`${fixed(robodojoMR0.value)}%`}</strong><span>RoboDojo avg. SR</span></div>
        </div>
      </section>

      <section className="project-section section-shell" id="data">
        <div className="section-heading">
          <h2>Data</h2>
        </div>
        <p className="section-intro">
          The training corpus combines broad visual-language supervision with
          curated robot trajectories across 12 embodiments. Vision-language data
          supplies semantic coverage, while the robot corpus combines RoboDesk,
          licensed real-robot data, and open simulation data to ground that
          knowledge in executable behavior.
        </p>

        <div className="subsection">
          <div className="subsection-copy">
            <h3>Vision-Language Data</h3>
            <p>
              {results.data.visionLanguageSamples / 1_000_000} million samples provide both general visual understanding and
              embodied grounding. General data accounts for 78% of the mixture,
              while embodied data contributes the remaining 22%. The embodied
              portion covers visual grounding, affordance grounding, scene
              captioning, and task planning. The general portion covers visual
              grounding and counting, general VQA, and spatial understanding.
            </p>
          </div>
          <figure className="paper-figure wide-figure">
            <img
              src="/assets/vision-language-data-20261030.svg?v=20261030-ppt"
              alt="Composition and examples of the vision-language dataset"
              width={2802}
              height={1208}
              loading="lazy"
            />
          </figure>
          <FigureViewer src="/assets/vision-language-data-20261030.svg?v=20261030-ppt" title="Vision-language data" />
        </div>

        <div className="subsection">
          <div className="subsection-copy">
            <h3>Robotic Manipulation Data</h3>
            <p>
              The final robot corpus contains {number(results.data.robotHours)} hours of retained
              tabletop trajectories across 12 embodiments. It comprises 3,235
              hours of curated open-source real-robot data from RoboDesk, 7,530
              hours of licensed real-robot data, and 4,295 hours of open
              simulation data. In total, 10,765 hours come from physical robot
              execution. The corpus is restricted to stationary-base manipulation
              and excludes segments containing base motion.
            </p>
          </div>
          <RobotDataFigure />
        </div>

      </section>

      <section className="project-section section-shell" id="method">
        <div className="section-heading">
          <h2>Method</h2>
        </div>
        <p className="section-intro">
          The method combines a three-stage training recipe with an event-driven
          agent. Discrete action-token pre-training establishes broad behavioral
          priors, continuous flow-matching mid-training transfers them to precise
          cross-embodiment control, and closed-loop post-training adapts the policy
          to errors encountered during real-world deployment.
        </p>

        <div className="subsection">
          <div className="subsection-copy">
            <h3>Training Recipe</h3>
            <p>
              The three stages separate broad multimodal learning from continuous
              control and real-world adaptation. SFT initializes a task policy
              for data collection. HG-DAgger, RECAP, and CFG-DAgger are alternative
              post-training strategies: each is trained anew from the same
              mid-training checkpoint using cumulative data, not sequentially
              applied to one policy.
            </p>
          </div>
          <div className="training-timeline" role="list" aria-label="Three-stage training timeline from scale to precision to reliability">
            <article className="training-stage" role="listitem">
              <header><span>01</span><b>Scale</b></header>
              <div className="training-stage-body">
                <small>Pre-training</small>
                <h4>FAST + VLM</h4>
                <p>Co-train vision-language supervision and discrete FAST action tokens to align multimodal features with robot behavior.</p>
                <strong className="stage-outcome">Broad action priors</strong>
              </div>
            </article>

            <article className="training-stage" role="listitem">
              <header><span>02</span><b>Precision</b></header>
              <div className="training-stage-body">
                <small>Mid-training</small>
                <h4>MMDiT + JiT</h4>
                <p>Replace discrete action prediction with continuous flow matching over a unified end-effector representation.</p>
                <strong className="stage-outcome">Coherent action chunks</strong>
              </div>
            </article>

            <article className="training-stage alignment-stage" role="listitem">
              <header><span>03</span><b>Reliability</b></header>
              <div className="training-stage-body">
                <small>Closed-loop post-training</small>
                <div className="alignment-steps" aria-label="Alternative post-training algorithms: HG-DAgger, RECAP, and CFG-DAgger">
                  <strong>HG-DAgger</strong><strong>RECAP</strong><strong>CFG-DAgger</strong>
                </div>
                <p>Learn from demonstrations and corrections. Rollback additionally retains replaced policy segments as negative supervision for CFG-DAgger.</p>
                <strong className="stage-outcome">Closed-loop policy optimization</strong>
              </div>
            </article>
          </div>
        </div>

        <div className="subsection" id="agent">
          <div className="subsection-copy">
            <h3>Event-Driven Agent</h3>
            <p>
              A Monitor, Memory Manager, and Planner operate above the VLA policy
              at different rates. The Monitor summarizes short visual windows,
              the Memory Manager converts them into persistent events with
              supporting keyframes, and the Planner is invoked only when an event
              changes task progress or feasibility. Low-level control therefore
              continues at robot frequency while the system preserves cross-task
              state and resumes objectives after temporary interruptions.
            </p>
          </div>
          {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- The horizontally scrollable diagram must be reachable by keyboard. */}
          <figure className="paper-figure wide-figure agent-figure" role="region" aria-label="Scrollable event-driven agent architecture" tabIndex={0}>
            <img
              src="/assets/embodied-agent-20261029.svg"
              alt="Event-driven agent with task context, an asynchronous agent loop, and high-frequency action execution"
              width={442}
              height={221}
              loading="lazy"
            />
          </figure>
          <FigureViewer src="/assets/embodied-agent-20261029.svg" title="Event-driven agent" />
        </div>
      </section>

      <section className="project-section section-shell" id="evaluation">
        <div className="section-heading">
          <h2>Evaluation</h2>
        </div>
        <p className="section-intro">
          Evaluation spans simulation benchmarks and physical robot deployment.
          Simulation measures standalone-policy transfer, while real-robot studies
          report success rate (SR) and progress score (PS) under matched
          post-training protocols.
        </p>

        <div className="subsection">
          <div className="subsection-copy">
            <h3>Simulation and Real-Robot Evaluation</h3>
            <p>
              We evaluate MR0 on RoboTwin 2.0, RoboDojo, and three real-robot
              tasks: letter-block placement, garment folding, and Ethernet
              insertion. The RoboDojo chart compares average success rates for
              selected methods.
            </p>
          </div>
          <RevealOnView className="result-card-grid">
            <article className="result-card">
              <header>
                <div><span>Simulation · Overall SR</span><h4>RoboTwin 2.0</h4></div>
                <strong>{`${robotwinMR0.value}%`}</strong>
              </header>
              <VerticalBarChart label={`RoboTwin 2.0 overall success-rate comparison, shown on an ${results.robotwin.min} to ${results.robotwin.max} percent scale`} min={results.robotwin.min} max={results.robotwin.max} ticks={[80, 90, 95]} items={results.robotwin.models} />
            </article>

            <article className="result-card">
              <header>
                <div><span>Simulation · Average SR (%)</span><h4>RoboDojo</h4></div>
                <strong>{`${fixed(robodojoMR0.value)}%`}</strong>
              </header>
              <VerticalBarChart label="RoboDojo average success rate (%) for selected methods" min={results.robodojo.min} max={results.robodojo.max} ticks={[0, 10, 20, 30, 40]} items={results.robodojo.models.map((model) => ({ ...model, display: fixed(model.value) }))} barTopValues />
            </article>

            <article className="result-card">
              <header>
                <div><span>Matched {results.realRobot.protocol} · Average</span><h4>Real Robot</h4></div>
                <MetricPair sr={results.realRobot.mr0.sr} ps={results.realRobot.mr0.ps} />
              </header>
              <GroupedVerticalChart label="Average real-robot success rate and progress score under matched HG-DAgger post-training, shown on a 60 to 100 percent scale" min={results.realRobot.min} max={results.realRobot.max} ticks={[0.6, 0.8, 1]} series={[results.realRobot.baseline.label, results.realRobot.mr0.label]} groups={[
                { label: "Avg. SR", bars: [{ label: results.realRobot.baseline.label, value: results.realRobot.baseline.sr }, { label: results.realRobot.mr0.label, value: results.realRobot.mr0.sr, highlight: true }] },
                { label: "Avg. PS", bars: [{ label: results.realRobot.baseline.label, value: results.realRobot.baseline.ps }, { label: results.realRobot.mr0.label, value: results.realRobot.mr0.ps, highlight: true }] },
              ]} />
            </article>

            <article className="result-card">
              <header>
                <div><span>Ethernet insertion · {results.ethernetRollback.protocol}</span><h4>{results.ethernetRollback.mr0.label}</h4></div>
                <MetricPair sr={results.ethernetRollback.mr0.sr} ps={results.ethernetRollback.mr0.ps} />
              </header>
              <GroupedVerticalChart label="CFG-DAgger compared with RECAP on Ethernet insertion using the same demonstrations and Rollback episodes" min={results.ethernetRollback.min} max={results.ethernetRollback.max} series={[results.ethernetRollback.baseline.label, results.ethernetRollback.mr0.label]} groups={[
                { label: "SR", bars: [{ label: results.ethernetRollback.baseline.label, value: results.ethernetRollback.baseline.sr, display: fixed(results.ethernetRollback.baseline.sr) }, { label: results.ethernetRollback.mr0.label, value: results.ethernetRollback.mr0.sr, display: fixed(results.ethernetRollback.mr0.sr), highlight: true }] },
                { label: "PS", bars: [{ label: results.ethernetRollback.baseline.label, value: results.ethernetRollback.baseline.ps, display: fixed(results.ethernetRollback.baseline.ps) }, { label: results.ethernetRollback.mr0.label, value: results.ethernetRollback.mr0.ps, display: fixed(results.ethernetRollback.mr0.ps), highlight: true }] },
              ]} />
            </article>

          </RevealOnView>
        </div>
      </section>

      <section className="project-section section-shell" id="demos">
        <div className="section-heading">
          <h2>Real-World Demos</h2>
        </div>
        <p className="section-intro">
          Real-robot demonstrations show the policy executing precise bimanual
          manipulation and the agent preserving task state across interruptions.
        </p>

        <div className="subsection">
          <div className="subsection-copy">
            <h3>Desktop Manipulation</h3>
            <p>
              Three desktop tasks cover ordered placement, deformable-object
              handling, and contact-rich insertion. The set stresses sequencing,
              bimanual coordination, recovery from local errors, and fine geometric
              alignment. Videos illustrate the tasks; the metrics summarize the
              paper&apos;s evaluations: HG-DAgger for letter-block placement and
              garment folding, and CFG-DAgger for Ethernet insertion. SR requires
              full completion; garment folding must finish within 120 seconds.
              Folding trials continue for up to 300 seconds, and later completion
              counts as failure for SR but retains its completion time for PS.
              PS measures consecutive correct placements for
              blocks, time-based execution efficiency for folding, and completed
              grasping, handover, and insertion milestones for Ethernet insertion.
            </p>
          </div>
          <div className="task-grid">
            <article className="task-card">
              <div className="task-media task-demo-media">
                <VideoPlayer
                  src="/video/letter-block-placement.mp4?v=20260828-letter1440"
                  poster="/assets/letter-block-placement-first-frame.jpg?v=20260828-letter1440"
                  ariaLabel="Letter-block Placement demo"
                />
              </div>
              <div className="task-copy">
                <h4>Letter-block Placement</h4>
                <dl><div><dt>SR</dt><dd>{fixed(results.demos.letterBlock.sr)}</dd></div><div><dt>PS</dt><dd>{fixed(results.demos.letterBlock.ps)}</dd></div></dl>
              </div>
            </article>
            <article className="task-card">
              <div className="task-media task-demo-media">
                <VideoPlayer
                  src="/video/garment-folding-demo.mp4?v=20260828-hq"
                  poster="/assets/garment-folding-first-frame.jpg"
                  ariaLabel="Garment Folding demo"
                />
              </div>
              <div className="task-copy">
                <h4>Garment Folding</h4>
                <dl><div><dt>SR</dt><dd>{fixed(results.demos.garment.sr)}</dd></div><div><dt>PS</dt><dd>{fixed(results.demos.garment.ps)}</dd></div></dl>
              </div>
            </article>
            <article className="task-card">
              <div className="task-media task-demo-media">
                <VideoPlayer
                  src="/video/network-cable-insertion.mp4?v=20260828-hq"
                  poster="/assets/network-cable-insertion-first-frame.jpg"
                  ariaLabel="Ethernet Insertion demo"
                />
              </div>
              <div className="task-copy">
                <h4>Ethernet Insertion</h4>
                <dl><div><dt>SR</dt><dd>{fixed(results.ethernetRollback.mr0.sr)}</dd></div><div><dt>PS</dt><dd>{fixed(results.ethernetRollback.mr0.ps)}</dd></div></dl>
              </div>
            </article>
          </div>
        </div>

        <div className="subsection" id="agent-demos">
          <div className="subsection-copy">
            <h3>Long-Horizon Agent Behaviors</h3>
            <p>
              Two deployment traces illustrate persistent event memory and
              conditional recovery. These videos are qualitative cases rather than
              an aggregate success-rate benchmark.
            </p>
          </div>
          <div className="agent-cases">
            <article>
              <figure className="agent-demo-media">
                <VideoPlayer
                  src="/video/vla_demo_01_en.mp4?v=20260828-hq"
                  poster="/assets/event-retrieval-first-frame.jpg"
                  ariaLabel="Cross-task Event Retrieval demo"
                />
              </figure>
              <div>
                <h4>Cross-task Event Retrieval</h4>
                <p>While the screwdriver subtask continues, a person moves the key from the bottom drawer to the middle drawer. The agent stores that event and later retrieves the key&apos;s last observed location for a new instruction.</p>
              </div>
            </article>
            <article>
              <figure className="agent-demo-media">
                <VideoPlayer
                  src="/video/vla_demo_02_en.mp4?v=20260828-hq"
                  poster="/assets/waiting-recovery-first-frame.jpg"
                  ariaLabel="Subtask Waiting and Recovery demo"
                />
              </figure>
              <div>
                <h4>Subtask Waiting and Recovery</h4>
                <p>When the key is temporarily removed, the agent preserves the pending placement objective and waits. Its return triggers reassessment, allowing the original instruction to resume and complete.</p>
              </div>
            </article>
          </div>
        </div>
      </section>

      <section className="project-section section-shell" id="conclusion">
        <div className="section-heading">
          <h2>Conclusion</h2>
        </div>
        <div className="conclusion-copy">
          <p>
            Meituan-Robotics-0 connects aligned heterogeneous data, continuous
            cross-embodiment control, closed-loop policy optimization, and
            event-level memory in one desktop-manipulation system. The evaluation
            shows that broad pre-training supports transfer, human interventions
            improve recovery from policy-induced states, Rollback pairs support
            CFG-DAgger for precision-sensitive insertion without value learning,
            and persistent events extend
            planning beyond the current observation.
          </p>
        </div>
      </section>

      <section className="project-section section-shell citation-section" id="citation">
        <div className="section-heading">
          <h2>Citation</h2>
        </div>
        <p className="section-intro">
          Please cite the technical report when using this work.
        </p>
        <div className="citation-block">
          <pre><code>{citationText}</code></pre>
          <CopyCitation text={citationText} />
        </div>
        <div className="citation-resources">
          <a className="resource primary" href="https://arxiv.org/abs/2611.08577">Technical Report ↗</a>
        </div>
      </section>

      <footer>
        <div className="section-shell footer-inner">
          <div className="brand"><span className="brand-mark">美团</span><span>Meituan Robotics</span></div>
          <span>Meituan-Robotics-0</span>
        </div>
      </footer>
    </main>
  );
}
