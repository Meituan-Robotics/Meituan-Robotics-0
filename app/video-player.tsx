"use client";

import { useId, useRef, useState } from "react";

type VideoPlayerProps = {
  src: string;
  poster: string;
  ariaLabel: string;
  className?: string;
};

const playbackRates = [1, 2, 3];

export default function VideoPlayer({
  src,
  poster,
  ariaLabel,
  className = "",
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const speedId = useId();
  const [playbackRate, setPlaybackRate] = useState(1);

  function changePlaybackRate(value: string) {
    const nextRate = Number(value);
    setPlaybackRate(nextRate);
    if (videoRef.current) {
      videoRef.current.playbackRate = nextRate;
      videoRef.current.defaultPlaybackRate = nextRate;
    }
  }

  return (
    <div className={`video-player${className ? ` ${className}` : ""}`}>
      <video
        ref={videoRef}
        controls
        playsInline
        preload="metadata"
        poster={poster}
        aria-label={ariaLabel}
      >
        <source src={src} type="video/mp4" />
      </video>
      <label className="playback-speed" htmlFor={speedId}>
        <span>Speed</span>
        <select
          id={speedId}
          value={playbackRate}
          onChange={(event) => changePlaybackRate(event.target.value)}
          aria-label={`${ariaLabel} playback speed`}
        >
          {playbackRates.map((rate) => (
            <option value={rate} key={rate}>{rate}×</option>
          ))}
        </select>
      </label>
    </div>
  );
}
