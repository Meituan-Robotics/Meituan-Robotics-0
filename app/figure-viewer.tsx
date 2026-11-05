"use client";

import "../public/figure-viewer.js";
import "./figure-viewer.css";

type FigureViewerProps = {
  src: string;
  title: string;
};

export default function FigureViewer({ src, title }: FigureViewerProps) {
  return (
    <div className="figure-viewer-action">
      <a
        className="figure-viewer-trigger"
        data-figure-zoom
        data-figure-title={title}
        href={src}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={`View ${title} at full size`}
        aria-haspopup="dialog"
      >
        <span aria-hidden="true">⤢</span>
        View full size
      </a>
    </div>
  );
}
