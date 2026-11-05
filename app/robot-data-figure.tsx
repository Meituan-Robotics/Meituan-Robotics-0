"use client";

import { useState, type KeyboardEvent } from "react";
import results from "../data/project-results.json";

type DataSource = "inhouse" | "open-real" | "open-sim";

const robotEmbodiments = [
  { label: "AgiBot G1", total: 3652, inhouse: 1255, openReal: 2192, openSim: 205 },
  { label: "Agilex Aloha", total: 2649, inhouse: 1000, openReal: 99, openSim: 1550 },
  { label: "Kuavo 4Pro", total: 2444, inhouse: 2249, openReal: 195, openSim: 0 },
  { label: "ARX Lift2", total: 1580, inhouse: 0, openReal: 0, openSim: 1580 },
  { label: "Galaxea R1", total: 1478, inhouse: 1478, openReal: 0, openSim: 0 },
  { label: "Franka Panda", total: 1015, inhouse: 0, openReal: 237, openSim: 778 },
  { label: "WheelLoong M1", total: 820, inhouse: 820, openReal: 0, openSim: 0 },
  { label: "Qinglong", total: 578, inhouse: 578, openReal: 0, openSim: 0 },
  { label: "Galaxea R1 Lite", total: 400, inhouse: 0, openReal: 400, openSim: 0 },
  { label: "AgiBot G2", total: 182, inhouse: 0, openReal: 0, openSim: 182 },
  { label: "Kuavo LB", total: 150, inhouse: 150, openReal: 0, openSim: 0 },
  { label: "RealMan Aida-L", total: 112, inhouse: 0, openReal: 112, openSim: 0 },
];

export default function RobotDataFigure() {
  const [hoveredSource, setHoveredSource] = useState<DataSource | null>(null);
  const [focusedSource, setFocusedSource] = useState<DataSource | null>(null);
  const [lockedSource, setLockedSource] = useState<DataSource | null>(null);
  const activeSource = hoveredSource ?? focusedSource ?? lockedSource;

  const clearHoveredSource = (source: DataSource) => {
    setHoveredSource((current) => (current === source ? null : current));
  };

  const clearFocusedSource = (source: DataSource) => {
    setFocusedSource((current) => (current === source ? null : current));
  };

  const toggleLockedSource = (source: DataSource) => {
    setFocusedSource(null);
    setLockedSource((current) => (current === source ? null : source));
  };

  const handleArcKeyDown = (
    event: KeyboardEvent<SVGCircleElement>,
    source: DataSource,
  ) => {
    if (!event.repeat && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      toggleLockedSource(source);
    }
  };

  return (
    <figure
      className="robot-data-figure"
      aria-label={`Composition of the ${results.data.robotHours.toLocaleString("en-US")}-hour robot dataset`}
      data-active-source={activeSource ?? undefined}
      onPointerLeave={() => setHoveredSource(null)}
    >
      <section className="donut-panel">
        <div className="donut-stage">
          <button
            type="button"
            className="donut-label open-sim-label source-control"
            data-source="open-sim"
            aria-pressed={lockedSource === "open-sim"}
            onPointerEnter={() => setHoveredSource("open-sim")}
            onPointerLeave={() => clearHoveredSource("open-sim")}
            onFocus={() => setFocusedSource("open-sim")}
            onBlur={() => clearFocusedSource("open-sim")}
            onClick={() => toggleLockedSource("open-sim")}
          >
            <span>Open Sim</span><strong>28.5%</strong>
          </button>
          <button
            type="button"
            className="donut-label open-real-label source-control"
            data-source="open-real"
            aria-pressed={lockedSource === "open-real"}
            onPointerEnter={() => setHoveredSource("open-real")}
            onPointerLeave={() => clearHoveredSource("open-real")}
            onFocus={() => setFocusedSource("open-real")}
            onBlur={() => clearFocusedSource("open-real")}
            onClick={() => toggleLockedSource("open-real")}
          >
            <span>RoboDesk</span><strong>21.5%</strong>
          </button>
          <button
            type="button"
            className="donut-label inhouse-label source-control"
            data-source="inhouse"
            aria-pressed={lockedSource === "inhouse"}
            onPointerEnter={() => setHoveredSource("inhouse")}
            onPointerLeave={() => clearHoveredSource("inhouse")}
            onFocus={() => setFocusedSource("inhouse")}
            onBlur={() => clearFocusedSource("inhouse")}
            onClick={() => toggleLockedSource("inhouse")}
          >
            <span>Licensed Real</span><strong>50.0%</strong>
          </button>
          <div
            className="source-donut"
            role="group"
            aria-label="50 percent licensed real, 21.5 percent RoboDesk, and 28.5 percent open simulation"
          >
            <div className="world-donut">
              <div className="donut-center"><strong>{results.data.robotHours.toLocaleString("en-US")}</strong><span>hours</span></div>
            </div>
            <svg
              className="donut-hit-map"
              viewBox="0 0 100 100"
              role="group"
              aria-label="Select a data source"
            >
              <circle
                className="donut-hit inhouse"
                data-source="inhouse"
                cx="50"
                cy="50"
                r="45"
                pathLength="100"
                fill="none"
                stroke="transparent"
                strokeWidth="10"
                strokeDasharray="49.6 50.4"
                strokeDashoffset="0"
                transform="rotate(-73 50 50)"
                pointerEvents="stroke"
                role="button"
                tabIndex={0}
                aria-label="Licensed Real, 50.0 percent"
                aria-pressed={lockedSource === "inhouse"}
                onPointerEnter={() => setHoveredSource("inhouse")}
                onPointerLeave={() => clearHoveredSource("inhouse")}
                onFocus={() => setFocusedSource("inhouse")}
                onBlur={() => clearFocusedSource("inhouse")}
                onClick={() => toggleLockedSource("inhouse")}
                onKeyDown={(event) => handleArcKeyDown(event, "inhouse")}
              />
              <circle
                className="donut-hit open-real"
                data-source="open-real"
                cx="50"
                cy="50"
                r="45"
                pathLength="100"
                fill="none"
                stroke="transparent"
                strokeWidth="10"
                strokeDasharray="20.9 79.1"
                strokeDashoffset="-50.2"
                transform="rotate(-73 50 50)"
                pointerEvents="stroke"
                role="button"
                tabIndex={0}
                aria-label="RoboDesk, 21.5 percent"
                aria-pressed={lockedSource === "open-real"}
                onPointerEnter={() => setHoveredSource("open-real")}
                onPointerLeave={() => clearHoveredSource("open-real")}
                onFocus={() => setFocusedSource("open-real")}
                onBlur={() => clearFocusedSource("open-real")}
                onClick={() => toggleLockedSource("open-real")}
                onKeyDown={(event) => handleArcKeyDown(event, "open-real")}
              />
              <circle
                className="donut-hit open-sim"
                data-source="open-sim"
                cx="50"
                cy="50"
                r="45"
                pathLength="100"
                fill="none"
                stroke="transparent"
                strokeWidth="10"
                strokeDasharray="27.7 72.3"
                strokeDashoffset="-71.7"
                transform="rotate(-73 50 50)"
                pointerEvents="stroke"
                role="button"
                tabIndex={0}
                aria-label="Open Sim, 28.5 percent"
                aria-pressed={lockedSource === "open-sim"}
                onPointerEnter={() => setHoveredSource("open-sim")}
                onPointerLeave={() => clearHoveredSource("open-sim")}
                onFocus={() => setFocusedSource("open-sim")}
                onBlur={() => clearFocusedSource("open-sim")}
                onClick={() => toggleLockedSource("open-sim")}
                onKeyDown={(event) => handleArcKeyDown(event, "open-sim")}
              />
            </svg>
          </div>
        </div>
        <div className="figure-legend world-legend"><span className="real">Real-world · 71.5%</span><span className="simulation">Simulation · 28.5%</span></div>
        <h4>Data-source Composition</h4>
      </section>

      <section className="embodiment-panel">
        <div className="stacked-chart" role="group" aria-label="Embodiment-wise duration in hours, split by source">
          {robotEmbodiments.map((robot) => (
            <div className="stacked-row" key={robot.label} aria-label={`${robot.label}: ${robot.total.toLocaleString("en-US")} hours`}>
              <span>{robot.label}</span>
              <div className="stacked-track">
                <i className="inhouse" data-source="inhouse" style={{ width: `${robot.inhouse / 40}%` }} />
                <i className="open-real" data-source="open-real" style={{ width: `${robot.openReal / 40}%` }} />
                <i className="open-sim" data-source="open-sim" style={{ width: `${robot.openSim / 40}%` }} />
              </div>
              <b>{robot.total.toLocaleString("en-US")}</b>
            </div>
          ))}
          <div className="stacked-axis" aria-hidden="true"><span /><div><i>0</i><i>1,000</i><i>2,000</i><i>3,000</i><i>4,000</i></div><span /></div>
          <p>Duration (hours)</p>
        </div>
        <div className="figure-legend source-legend">
          <button
            type="button"
            className="inhouse source-control"
            data-source="inhouse"
            aria-pressed={lockedSource === "inhouse"}
            onPointerEnter={() => setHoveredSource("inhouse")}
            onPointerLeave={() => clearHoveredSource("inhouse")}
            onFocus={() => setFocusedSource("inhouse")}
            onBlur={() => clearFocusedSource("inhouse")}
            onClick={() => toggleLockedSource("inhouse")}
          >
            Licensed Real
          </button>
          <button
            type="button"
            className="open-real source-control"
            data-source="open-real"
            aria-pressed={lockedSource === "open-real"}
            onPointerEnter={() => setHoveredSource("open-real")}
            onPointerLeave={() => clearHoveredSource("open-real")}
            onFocus={() => setFocusedSource("open-real")}
            onBlur={() => clearFocusedSource("open-real")}
            onClick={() => toggleLockedSource("open-real")}
          >
            RoboDesk
          </button>
          <button
            type="button"
            className="open-sim source-control"
            data-source="open-sim"
            aria-pressed={lockedSource === "open-sim"}
            onPointerEnter={() => setHoveredSource("open-sim")}
            onPointerLeave={() => clearHoveredSource("open-sim")}
            onFocus={() => setFocusedSource("open-sim")}
            onBlur={() => clearFocusedSource("open-sim")}
            onClick={() => toggleLockedSource("open-sim")}
          >
            Open Sim
          </button>
        </div>
        <h4>Embodiment-wise Duration</h4>
      </section>
    </figure>
  );
}
